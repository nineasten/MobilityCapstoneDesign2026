#!/usr/bin/env python3

import os
import re
import sys
import termios
import tty
from pathlib import Path
from threading import Lock

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


IMAGE_TOPIC = "/arducam_imx219/image_raw"
OUTPUT_DIR = Path("/home/vic/colmap/study_workspace/images")
FILENAME_PATTERN = re.compile(r"^image(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


class FrameBuffer:
    def __init__(self) -> None:
        self._bridge = CvBridge()
        self._frame = None
        self._lock = Lock()

    def callback(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("Failed to convert frame: %s", exc)
            return

        with self._lock:
            self._frame = frame

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()


def next_image_path(output_dir: Path) -> Path:
    max_index = 0
    for path in output_dir.iterdir():
        match = FILENAME_PATTERN.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return output_dir / f"image{max_index + 1:04d}.jpg"


def read_single_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rospy.init_node("capture_to_colmap", anonymous=True)
    frame_buffer = FrameBuffer()
    rospy.Subscriber(IMAGE_TOPIC, Image, frame_buffer.callback, queue_size=1)

    print(f"Listening on {IMAGE_TOPIC}")
    print(f"Saving images to {OUTPUT_DIR}")
    print("Press p to save the latest frame. Press q to quit.")

    while not rospy.is_shutdown():
        key = read_single_key()
        if key.lower() == "q":
            print("\nExiting.")
            return 0

        if key.lower() != "p":
            continue

        frame = frame_buffer.get_frame()
        if frame is None:
            print("\nNo frame received yet. Wait a moment and try again.")
            continue

        output_path = next_image_path(OUTPUT_DIR)
        if not cv2.imwrite(str(output_path), frame):
            print(f"\nFailed to save {output_path}")
            continue

        print(f"\nSaved {output_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
