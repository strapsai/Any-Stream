import numpy as np
import torch
from typing import List
from .predictions import Predictions
from mapanything.utils.geometry import depthmap_to_world_frame

class MapAnythingAdapter:
    """
    Adapter to use MapAnything model within DA3-Streaming pipeline.
    Converts MapAnything outputs to the unified Predictions format.
    """

    def __init__(self, device: str = "cuda", checkpoint_path: str = None):
        self.device = device
        self.model = None
        self.checkpoint_path = checkpoint_path
        self._k_prior_logged = False

    def load(self):
        """Load MapAnything from a local directory or the HF hub.

        PyTorchModelHubMixin.from_pretrained accepts either, so checkpoint_path
        can be a directory holding config.json + model.safetensors. Prefer that
        on deployment: the hub route needs network access at model-load time.
        """
        from mapanything.models import MapAnything

        source = self.checkpoint_path or "facebook/map-anything"
        self.model = MapAnything.from_pretrained(source)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"MapAnything model loaded from {source}.")

    def _scaled_K(self, intrinsics, ref_image_path, views):
        """K for the on-disk images, mapped to the resolution load_images() produced.

        load_images() runs crop_resize_if_necessary(), and
        preprocess_input_views_for_inference() then builds the rays from the
        RESIZED shape -- it never rescales intrinsics. A K left at the on-disk
        resolution therefore describes a different image than the one being
        encoded: no error, just quietly wrong geometry.

        Reuses MapAnything's own crop_resize_if_necessary() rather than scaling by
        hand: the resize is isotropic plus a crop, so a per-axis scale gets fx/fy
        and the principal point wrong (~9 px on this data).
        """
        from PIL import Image
        from mapanything.utils.cropping import crop_resize_if_necessary

        Hp, Wp = views[0]["img"].shape[-2:]
        img = Image.open(ref_image_path)
        Wo, Ho = img.size

        K_in = np.asarray(intrinsics, dtype=np.float64)
        # returns (image, intrinsics) when only intrinsics is passed alongside
        K_out = np.asarray(crop_resize_if_necessary(
            image=img, resolution=(Wp, Hp), intrinsics=K_in)[1])

        if not self._k_prior_logged:
            print(f"[MapAnythingAdapter] intrinsics prior: on-disk {Wo}x{Ho} -> "
                  f"model {Wp}x{Hp}\n  effective K:\n{K_out}")
            self._k_prior_logged = True
        return torch.as_tensor(K_out, dtype=torch.float32)

    def infer(self, image_paths: List[str], intrinsics=None) -> Predictions:
        """
        Run inference and return unified Predictions object.

        Args:
            image_paths: List of paths to images
            intrinsics: Optional 3x3 K (numpy or torch) for the images ON DISK.
                Rescaled here to the resolution load_images() resized them to,
                then attached to every view as a prior.

        Returns:
            Predictions object with W2C extrinsics
        """
        from mapanything.utils.image import load_images

        # 1. Load images
        views = load_images(image_paths)
        print(f"Loaded {len(views)} views")

        if intrinsics is not None:
            # Every frame in a chunk shares a resolution and target, so map K once.
            K_model = self._scaled_K(intrinsics, image_paths[0], views)
            for v in views:
                # (1, 3, 3), not (3, 3): get_rays_in_camera_frame() mirrors the
                # batching of its input, and every other field in the view carries
                # B=1. An unbatched K yields (H, W, 3) rays, which the model then
                # fails to index with its per-sample mask of shape (1,).
                v["intrinsics"] = K_model[None]

        # 2. Run inference
        with torch.no_grad():
            outputs = self.model.infer(views,
                                    #    use_multiview_confidence=True,
                                    #    confidence_percentile=10,
                                    #    apply_confidence_mask=True,
                                    # # apply_mask=True, # Apply masking to dense geometry outputs
                                    # mask_edges=True,
                                )

        print("Inference complete!")

        depths = []
        confs = []
        extrinsics = []
        intrinsics = []
        images_out = []
        masks = []
        world_points_list = []

        for view_idx, pred in enumerate(outputs):
            # Extract data from predictions
            depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
            depth = depthmap_torch.cpu().numpy()
            camera_pose_c2w_torch = pred["camera_poses"][0]  # (4, 4)
            intrinsic_torch = pred["intrinsics"][0]  # (3, 3)

            # Compute world points and valid_mask from depth (same as demo_images_only_inference.py)
            pts3d_computed, valid_mask = depthmap_to_world_frame(
                depthmap_torch, intrinsic_torch, camera_pose_c2w_torch
            )

            camera_pose_c2w = camera_pose_c2w_torch.cpu().numpy()
            intrinsic = intrinsic_torch.cpu().numpy()

            img_no_norm = pred["img_no_norm"][0].cpu().numpy() # (H, W, 3)
            pred_mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

            # Combine prediction mask with valid depth mask
            mask = pred_mask & valid_mask.cpu().numpy()

            # convert extrinsics from C2W to W2C
            camera_pose_w2c = np.linalg.inv(camera_pose_c2w) # (4, 4)
            camera_pose_w2c = camera_pose_w2c[:3, :] # (3, 4)

            if img_no_norm.max() <= 1.0:
                img = (img_no_norm*255).astype(np.uint8)
            else:
                img = img_no_norm.astype(np.uint8)

            # Get confidence scores
            conf = pred["conf"][0].cpu().numpy()  # (H, W)
            # conf = (conf - conf.min()) / (conf.max() - conf.min()) * 100.0 #TODO: Implement cross view confidence.
            
            depths.append(depth)
            confs.append(conf)
            extrinsics.append(camera_pose_w2c)
            intrinsics.append(intrinsic)
            images_out.append(img)
            masks.append(mask)
            world_points_list.append(pts3d_computed.cpu().numpy())  # (H, W, 3)
        
         # Stack arrays
        conf_array = np.stack(confs)    # (N, H, W)
        mask_array = np.stack(masks)    # (N, H, W)

        # Normalize confidence across all views to 0-100 range
        conf_min, conf_max = conf_array.min(), conf_array.max()
        if conf_max > conf_min:
            conf_array = (conf_array - conf_min) / (conf_max - conf_min) * 100.0
        else:
            conf_array = np.full_like(conf_array, 50.0)

        # Set confidence to zero for invalid mask regions
        conf_array[~mask_array] = 0.0


        return Predictions(
            depth=np.stack(depths),           # (N, H, W)
            conf=conf_array,             # (N, H, W)
            extrinsics=np.stack(extrinsics),  # (N, 3, 4)
            intrinsics=np.stack(intrinsics),  # (N, 3, 3)
            processed_images=np.stack(images_out),  # (N, H, W, 3)
            mask=np.stack(masks),             # (N, H, W) bool
            world_points=np.stack(world_points_list),  # (N, H, W, 3)
        )