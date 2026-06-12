#!/usr/bin/env python3

import shutil
import subprocess

import rospy


def optional_int_param(name):
    value = rospy.get_param(name, "")
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def run_v4l2_control(video_device, control_name, value):
    command = [
        "v4l2-ctl",
        "-d",
        video_device,
        "-c",
        f"{control_name}={value}",
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown error"
        rospy.logwarn("Failed to set %s=%s on %s: %s", control_name, value, video_device, detail)
        return False

    rospy.loginfo("Applied camera control %s=%s on %s", control_name, value, video_device)
    return True


def main():
    rospy.init_node("arducam_camera_config")

    video_device = rospy.get_param("~video_device")
    auto_exposure = optional_int_param("~auto_exposure")
    exposure_time_absolute = optional_int_param("~exposure_time_absolute")
    exposure_dynamic_framerate = optional_int_param("~exposure_dynamic_framerate")
    gain = optional_int_param("~gain")
    white_balance_automatic = optional_int_param("~white_balance_automatic")
    white_balance_temperature = optional_int_param("~white_balance_temperature")
    power_line_frequency = optional_int_param("~power_line_frequency")
    brightness = optional_int_param("~brightness")
    contrast = optional_int_param("~contrast")
    saturation = optional_int_param("~saturation")

    if shutil.which("v4l2-ctl") is None:
        rospy.logerr("v4l2-ctl is not installed, so camera controls cannot be configured.")
        return

    if auto_exposure is not None:
        run_v4l2_control(video_device, "auto_exposure", auto_exposure)
    if exposure_dynamic_framerate is not None:
        run_v4l2_control(video_device, "exposure_dynamic_framerate", exposure_dynamic_framerate)

    if auto_exposure == 1 and exposure_time_absolute is not None:
        run_v4l2_control(video_device, "exposure_time_absolute", exposure_time_absolute)
    if gain is not None:
        run_v4l2_control(video_device, "gain", gain)
    if white_balance_automatic is not None:
        run_v4l2_control(video_device, "white_balance_automatic", white_balance_automatic)
    if white_balance_automatic == 0 and white_balance_temperature is not None:
        run_v4l2_control(video_device, "white_balance_temperature", white_balance_temperature)
    if power_line_frequency is not None:
        run_v4l2_control(video_device, "power_line_frequency", power_line_frequency)
    if brightness is not None:
        run_v4l2_control(video_device, "brightness", brightness)
    if contrast is not None:
        run_v4l2_control(video_device, "contrast", contrast)
    if saturation is not None:
        run_v4l2_control(video_device, "saturation", saturation)


if __name__ == "__main__":
    main()
