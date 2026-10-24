"""Local ground evidence in an ENU map, independent of ROS and map ownership.

A spatial median grid prevents a dense roof from winning by point count. A
bounded-slope, low consensus plane rejects roofs and isolated low outliers.
This is geometric evidence, not a semantic ground classifier: a map containing
only a broad roof can still call that roof ground.
"""
from dataclasses import dataclass, fields
import math
import numpy as np


@dataclass(frozen=True)
class GroundConfig:
    enabled: bool = False
    cell_m: float = 4.0
    min_points_per_cell: int = 40
    max_cell_height_span_m: float = 2.0
    min_total_cells: int = 60
    radius_m: float = 20.0
    min_support_cells: int = 6
    plane_band_m: float = 0.5
    max_slope: float = 0.35
    max_below_fraction: float = 0.10
    min_spread_m: float = 2.0
    max_support_distance_m: float = 12.0
    hypotheses: int = 128

    def __post_init__(self):
        for name in ('cell_m', 'radius_m', 'plane_band_m', 'max_slope',
                     'min_spread_m', 'max_support_distance_m', 'max_cell_height_span_m'):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f'GroundLevel.{name} must be finite and positive')
        for name in ('min_points_per_cell', 'min_total_cells', 'min_support_cells', 'hypotheses'):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1:
                raise ValueError(f'GroundLevel.{name} must be a positive integer')
        if self.min_support_cells < 3:
            raise ValueError('GroundLevel.min_support_cells must be at least 3')
        if not 0 <= self.max_below_fraction < 0.5:
            raise ValueError('GroundLevel.max_below_fraction must be in [0, 0.5)')
        if not isinstance(self.enabled, bool):
            raise ValueError('GroundLevel.enabled must be boolean')

    @classmethod
    def from_dict(cls, value=None):
        value = value or {}
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown GroundLevel settings: {sorted(unknown)}')
        return cls(**value)


class GroundSurface:
    """Read-only grid belonging to exactly one transformed map snapshot."""
    def __init__(self, points, config=None):
        self.config = config if isinstance(config, GroundConfig) else GroundConfig.from_dict(config)
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('ground points must have shape (N, 3)')
        p = p[np.isfinite(p).all(axis=1)]
        self.input_points = len(p)
        self.rejected_mixed_cells = 0
        self.cells = np.empty((0, 3), dtype=float)
        if self.config.enabled and len(p):
            keys = np.floor(p[:, :2] / self.config.cell_m).astype(np.int64)
            _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            order = np.argsort(inverse, kind='stable')
            offsets = np.r_[0, np.cumsum(counts)]
            # Median XY, not cell centres: a sloped plane remains exact even
            # when a partially observed cell is sampled on only one edge.
            cells = []
            for i in np.flatnonzero(counts >= self.config.min_points_per_cell):
                sample = p[order[offsets[i]:offsets[i+1]]]
                lo, hi = np.percentile(sample[:, 2], [10, 90])
                if hi - lo > self.config.max_cell_height_span_m:
                    # Two equally populated slabs must not invent a ground
                    # halfway between them. Walls and vertically mixed cells
                    # supply no ground vote; the original cloud is untouched.
                    self.rejected_mixed_cells += 1
                    continue
                cells.append(np.median(sample, axis=0))
            self.cells = np.asarray(cells).reshape(-1, 3)
        self.cells.setflags(write=False)

    def query(self, x, y):
        cfg = self.config
        out = dict(ground_z=None, status='disabled' if not cfg.enabled else 'insufficient_map',
                   total_cells=len(self.cells), rejected_mixed_cells=self.rejected_mixed_cells,
                   nearby_cells=0, support_cells=0)
        if not cfg.enabled or len(self.cells) < cfg.min_total_cells:
            return out
        if not np.isfinite([x, y]).all():
            return dict(out, status='invalid_query')
        delta = self.cells[:, :2] - [x, y]
        near = np.einsum('ij,ij->i', delta, delta) <= cfg.radius_m**2
        xy, z = delta[near], self.cells[near, 2]
        out['nearby_cells'] = len(z)
        if len(z) < cfg.min_support_cells:
            return dict(out, status='insufficient_local_cells')
        A = np.column_stack((xy, np.ones(len(z))))
        # Fixed seed makes repeated/revised-query diagnostics reproducible.
        rng = np.random.default_rng(0)
        low = np.flatnonzero(z <= np.percentile(z, 25))
        candidates = []
        for indices in (low, np.arange(len(z))):
            if len(indices) >= 3:
                sol, _, rank, _ = np.linalg.lstsq(A[indices], z[indices], rcond=None)
                if rank == 3:
                    candidates.append(sol)
        for k in range(cfg.hypotheses):
            pool = low if k < cfg.hypotheses // 2 and len(low) >= 3 else np.arange(len(z))
            idx = rng.choice(pool, 3, replace=False)
            if abs(np.linalg.det(A[idx])) > 1e-6:
                candidates.append(np.linalg.solve(A[idx], z[idx]))
        best = None
        for sol in candidates:
            if np.linalg.norm(sol[:2]) > cfg.max_slope:
                continue
            residual = z - A @ sol
            support = np.abs(residual) <= cfg.plane_band_m
            below = int((residual < -cfg.plane_band_m).sum())
            n = int(support.sum())
            if n < cfg.min_support_cells or below > cfg.max_below_fraction * len(z):
                continue
            score = n - 3 * below
            key = (score, -float(np.median(np.abs(residual[support]))))
            if best is None or key > best[0]:
                best = (key, support, sol)
        if best is None:
            return dict(out, status='no_low_plane')
        support = best[1]
        for _ in range(3):
            sol, _, rank, _ = np.linalg.lstsq(A[support], z[support], rcond=None)
            residual = z - A @ sol
            new = np.abs(residual) <= cfg.plane_band_m
            if np.array_equal(new, support) or new.sum() < cfg.min_support_cells:
                break
            support = new
        # Refit the final mask and validate; no percentile fallback on a line.
        sol, _, rank, _ = np.linalg.lstsq(A[support], z[support], rcond=None)
        residual = z - A @ sol
        spread = np.linalg.svd(xy[support] - xy[support].mean(0), compute_uv=False) / np.sqrt(support.sum())
        slope = float(np.linalg.norm(sol[:2]))
        below = int((residual < -cfg.plane_band_m).sum())
        nearest = float(np.min(np.linalg.norm(xy[support], axis=1)))
        out.update(support_cells=int(support.sum()), slope=slope,
                   residual_rmse_m=float(np.sqrt(np.mean(residual[support]**2))),
                   below_cells=below, nearest_support_m=nearest)
        if rank < 3 or spread[-1] < cfg.min_spread_m:
            return dict(out, status='poor_spatial_support')
        if slope > cfg.max_slope or below > cfg.max_below_fraction * len(z):
            return dict(out, status='unstable_plane')
        if nearest > cfg.max_support_distance_m:
            return dict(out, status='ground_too_far')
        return dict(out, status='supported', ground_z=float(sol[2]), plane_dx=float(sol[0]), plane_dy=float(sol[1]))
