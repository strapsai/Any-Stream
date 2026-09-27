"""One isolated numerical solve. Invoked by the ROS coordinator without a shell."""
import argparse
import platform

import numpy as np
import scipy

from .metric_io import read_bundle, validate_graph, validate_poses, write_bundle
from .metric_surface import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request, metadata = read_bundle(args.request)
    graph = validate_graph(request["graph"])
    poses, diagnostics = solve(graph,
                               **request["parameters"],
                               initial_absolutes=request.get("initial_absolutes"))
    validate_poses(poses)
    diagnostics["environment"] = dict(python=platform.python_version(),
                                      numpy=np.__version__,
                                      scipy=scipy.__version__)
    write_bundle(args.output, dict(absolutes=poses, diagnostics=diagnostics), metadata)


if __name__ == "__main__":
    main()
