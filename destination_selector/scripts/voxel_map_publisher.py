#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_PUBLISHER = Path("/home/vic/colmap/python/examples/publish_voxel_map_ros.py")


def strip_ros_remap_args(args: list[str]) -> list[str]:
    return [arg for arg in args if ":=" not in arg]


def main() -> None:
    publisher = Path(
        os.environ.get("VOXEL_MAP_ROS_PUBLISHER", str(DEFAULT_PUBLISHER))
    )
    if not publisher.exists():
        print(
            f"Voxel map ROS publisher not found: {publisher}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.execvp(
        "python3",
        ["python3", str(publisher), *strip_ros_remap_args(sys.argv[1:])],
    )


if __name__ == "__main__":
    main()
