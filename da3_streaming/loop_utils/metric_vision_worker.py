"""Persistent local GPU worker for ROS environments with a separate model Python.

Line-delimited JSON control and numeric NPZ payloads; stdout is protocol-only.
No downloads, dependency installation, ROS imports or database writes occur.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np


def sample(array, points):
    import cv2
    return cv2.remap(array.astype(np.float32),
                     points[:, 0].astype(np.float32).reshape(-1, 1),
                     points[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0).reshape(len(points), -1)


def load_ufm(path):
    import torch
    from mapanything.models.external.dinov2.hub import backbones
    from uniflowmatch.models.ufm import UniFlowMatchConfidence
    original = torch.hub.load

    def local(repo, model, *args, **kwargs):
        if repo != 'facebookresearch/dinov2':
            raise RuntimeError('unsupported offline architecture request: ' + repo)
        return getattr(backbones, model)(pretrained=False)

    torch.hub.load = local
    try:
        return UniFlowMatchConfidence.from_pretrained(str(path), strict=True).eval().cuda()
    finally:
        torch.hub.load = original


def ufm_pair(model, A, B):
    import cv2, torch

    def infer(a, b):
        # The measured reference implementation needs FP32 here. The override
        # is confined to this dedicated process and restored after each call.
        previous = torch.autocast
        torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
        try:
            with torch.inference_mode():
                p = model.predict_correspondences_batched(
                    torch.from_numpy(np.ascontiguousarray(a)).cuda(),
                    torch.from_numpy(np.ascontiguousarray(b)).cuda())
            return p.flow.flow_output[0].cpu().numpy().transpose(
                1, 2, 0), p.covisibility.mask[0].cpu().numpy()
        finally:
            torch.autocast = previous

    f, c = infer(A, B)
    h, w = c.shape
    yy, xx = np.meshgrid(np.arange(2, h, 4), np.arange(2, w, 4), indexing='ij')
    a = np.c_[xx.ravel(), yy.ravel()].astype(float)
    flow = sample(f, a)
    b = a + flow
    confidence = sample(c, a)[:, 0]
    valid = (confidence > .9) & (b[:, 0] >= 2) & (b[:, 1] >= 2) & (b[:, 0] < B.shape[1] -
                                                                   3) & (b[:, 1] < B.shape[0] - 3)
    a, b, flow, confidence = a[valid], b[valid], flow[valid], confidence[valid]
    if len(a) >= 30:
        g, d = infer(B, A)
        cycle = np.linalg.norm(flow + sample(g, b), axis=1)
        good = (sample(d, b)[:, 0] > .9) & (cycle < 1.5)
        a, b, confidence = a[good], b[good], confidence[good]
    else:
        a, b, confidence = a[:0], b[:0], confidence[:0]
    gate = 'insufficient_image_matches'
    reciprocal = len(a)
    if len(a) >= 30:
        _, mask = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, 1.5, .999, 10000)
        if mask is not None:
            good = mask.ravel().astype(bool)
            a, b, confidence = a[good], b[good], confidence[good]
            if len(a) >= 30 and min(*np.ptp(a, axis=0), *np.ptp(b, axis=0)) > 15: gate = 'accepted'
            else: gate = 'insufficient_2d_geometry'
        else: gate = 'epipolar_fit_failed'
    if len(a) > 1500:
        ids = np.linspace(0, len(a) - 1, 1500, dtype=int)
        a, b, confidence = a[ids], b[ids], confidence[ids]
    return dict(a=a, b=b, confidence=confidence), dict(gate=gate,
                                                       image_matches=len(a),
                                                       reciprocal_matches=reciprocal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['streamer', 'ufm'], required=True)
    ap.add_argument('--config')
    ap.add_argument('--root', required=True)
    ap.add_argument('--ros-module-dir')
    ap.add_argument('--checkpoint')
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def reply(row):
        protocol.write(json.dumps(row, allow_nan=False) + '\n')
        protocol.flush()

    try:
        import torch, cv2
        torch.set_num_threads(4)
        cv2.setNumThreads(4)
        start = time.monotonic()
        if args.mode == 'streamer':
            import yaml
            sys.path.insert(0, args.ros_module_dir)
            from anystreaming import AnyStreamer
            streamer = AnyStreamer(yaml.safe_load(Path(args.config).read_text()))
        else:
            model = load_ufm(args.checkpoint)
        torch.cuda.synchronize()
        reply(
            dict(event='ready',
                 mode=args.mode,
                 model_load_seconds=time.monotonic() - start,
                 python=sys.version,
                 torch=torch.__version__,
                 gpu=torch.cuda.get_device_name()))
        for line in sys.stdin:
            request = json.loads(line)
            ident = str(request['id'])
            if request.get('operation') == 'close': break
            if not ident.replace('_', '').isalnum(): raise ValueError('invalid request ID')
            try:
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                if args.mode == 'streamer':
                    streamer.process_chunk(request['image_paths'],
                                           final=bool(request.get('final', False)))
                    observation = streamer.metric_last_observation
                    pred = observation['predictions']
                    arrays = {
                        key: np.asarray(getattr(pred, key))
                        for key in ('depth', 'conf', 'mask', 'extrinsics', 'intrinsics',
                                    'processed_images', 'world_points')
                    }
                    s, R, t = observation['visual_pose']
                    details = dict(visual_pose=[float(s), R.tolist(),
                                                t.tolist()],
                                   chunk_index=streamer.chunk_idx - 1)
                else:
                    with np.load(request['input_npz'], allow_pickle=False) as x:
                        A = x['a'].copy()
                        B = x['b'].copy()
                    arrays, details = ufm_pair(model, A, B)
                torch.cuda.synchronize()
                compute_seconds = time.monotonic() - start
                result = root / (ident + '.npz')
                temporary = root / (ident + '.tmp.npz')
                np.savez(temporary, **arrays)
                temporary.replace(result)
                reply(
                    dict(event='result',
                         id=ident,
                         path=str(result),
                         sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                         compute_seconds=compute_seconds,
                         wall_seconds=time.monotonic() - start,
                         gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                         gpu_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                         **details))
            except Exception as exc:
                traceback.print_exc()
                reply(dict(event='error', id=ident, reason=repr(exc)))
    except Exception as exc:
        traceback.print_exc()
        reply(dict(event='fatal', reason=repr(exc)))


if __name__ == '__main__': main()
