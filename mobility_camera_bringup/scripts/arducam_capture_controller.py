#!/usr/bin/env python3

import os
import select
import termios
import threading
import time
import tty

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ArducamCaptureController:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.camera_name = rospy.get_param("~camera_name", "arducam_imx219")
        self.image_topic = rospy.get_param("~image_topic", f"/{self.camera_name}/image_raw")
        self.capture_interval = float(rospy.get_param("~capture_interval", 0.5))
        self.image_prefix = rospy.get_param("~image_prefix", "arducam")
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 95))

        default_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "captured_images",
        )
        save_root_param = rospy.get_param("~save_root", default_root)
        self.save_root = os.path.abspath(os.path.expanduser(save_root_param))

        session_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_root, f"{self.camera_name}_{session_stamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.capture_enabled = False
        self.capture_count = 0
        self.last_saved_at = 0.0
        self.latest_frame = None
        self.latest_stamp = None

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self.capture_timer = rospy.Timer(rospy.Duration(0.1), self.capture_timer_callback)
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        rospy.loginfo("Arducam capture controller is ready.")
        rospy.loginfo("Capture folder: %s", self.session_dir)
        rospy.loginfo("Press 's' to start saving images every %.1f seconds.", self.capture_interval)
        rospy.loginfo("Press 'e' to end auto capture.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as err:
            rospy.logerr_throttle(5.0, "Failed to convert image: %s", err)
            return

        with self.lock:
            self.latest_frame = frame
            self.latest_stamp = msg.header.stamp

    def capture_timer_callback(self, _event):
        if not self.capture_enabled:
            return

        now = time.time()
        if now - self.last_saved_at < self.capture_interval:
            return

        with self.lock:
            if self.latest_frame is None:
                rospy.logwarn_throttle(5.0, "Capture is enabled, but no image frame has arrived yet.")
                return

            frame = self.latest_frame.copy()
            stamp = self.latest_stamp

        image_index = self.capture_count + 1
        filename = self.build_filename(image_index, stamp, now)
        image_path = os.path.join(self.session_dir, filename)
        saved = cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not saved:
            rospy.logerr("Failed to save image %d to %s", image_index, image_path)
            return

        self.capture_count = image_index
        self.last_saved_at = now
        rospy.loginfo("Saved image %d: %s", self.capture_count, image_path)

    def build_filename(self, image_index, stamp, fallback_time):
        if stamp is not None and not stamp.is_zero():
            secs = stamp.secs
            nsecs = stamp.nsecs
        else:
            secs = int(fallback_time)
            nsecs = int((fallback_time - secs) * 1e9)

        return f"{self.image_prefix}_{image_index:05d}_{secs}_{nsecs:09d}.jpg"

    def keyboard_loop(self):
        try:
            tty_fd, tty_source = self.open_keyboard_fd()
        except OSError as err:
            rospy.logwarn("Keyboard control is unavailable because no usable terminal was found: %s", err)
            return

        original_settings = termios.tcgetattr(tty_fd)
        try:
            tty.setcbreak(tty_fd)
            rospy.loginfo("Keyboard control is listening on %s.", tty_source)
            while not rospy.is_shutdown():
                readable, _, _ = select.select([tty_fd], [], [], 0.2)
                if not readable:
                    continue

                key = os.read(tty_fd, 1).decode("utf-8", errors="ignore").lower()
                if key == "s":
                    self.start_capture()
                elif key == "e":
                    self.stop_capture()
        finally:
            termios.tcsetattr(tty_fd, termios.TCSADRAIN, original_settings)
            os.close(tty_fd)

    def open_keyboard_fd(self):
        last_error = None

        if os.isatty(0):
            return os.dup(0), "stdin"

        candidate_paths = [
            f"/proc/{os.getppid()}/fd/0",
            "/dev/tty",
        ]

        for candidate_path in candidate_paths:
            try:
                candidate_fd = os.open(candidate_path, os.O_RDONLY | os.O_NOCTTY)
                if not os.isatty(candidate_fd):
                    os.close(candidate_fd)
                    continue
                return candidate_fd, candidate_path
            except OSError as err:
                last_error = err

        if last_error is None:
            last_error = OSError("TTY source not found")
        raise last_error

    def start_capture(self):
        if self.capture_enabled:
            rospy.loginfo("Auto capture is already running. Saved images so far: %d", self.capture_count)
            return

        self.capture_enabled = True
        self.last_saved_at = 0.0
        rospy.loginfo("Auto capture started. Images will be saved every %.1f seconds.", self.capture_interval)

    def stop_capture(self):
        if not self.capture_enabled:
            rospy.loginfo("Auto capture is already stopped. Total saved images: %d", self.capture_count)
            return

        self.capture_enabled = False
        rospy.loginfo("Auto capture stopped. Total saved images: %d", self.capture_count)


def main():
    rospy.init_node("arducam_capture_controller")
    controller = ArducamCaptureController()
    rospy.on_shutdown(controller.stop_capture)
    rospy.spin()


if __name__ == "__main__":
    main()
