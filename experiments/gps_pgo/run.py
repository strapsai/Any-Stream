"""Run the selected experimental objective on a portable numeric graph file."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import optimizer, reference
from .problem import load_problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=150)
    args = parser.parse_args()
    data = load_problem(args.problem)
    poses, diagnostics = (reference if args.reference else optimizer).solve(
        data, max_nfev=args.max_nfev)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "transforms.npz",
                        scales=np.array([x[0] for x in poses]),
                        rotations=np.array([x[1] for x in poses]),
                        translations=np.array([x[2] for x in poses]))
    errors = np.concatenate([
        optimizer.transform(a[1::2], poses[k]) - optimizer.transform(b[1::2], poses[k + 1])
        for k, (a, b) in enumerate(data["seams"])
    ])
    diagnostics["heldout_adjacent_p90_m"] = float(np.quantile(np.linalg.norm(errors, axis=1), .9))
    text = json.dumps(diagnostics, indent=2, allow_nan=False)
    (args.output / "solver.json").write_text(text + "\n")
    print(text)
    if not diagnostics["success"]:
        raise SystemExit("Optimizer did not converge; outputs are diagnostic, not an accepted result")


if __name__ == "__main__":
    main()
