#!/usr/bin/env python3
"""
@dead_pixels.py

removes dead pixels from bayer images
"""

from ximea import xiapi
from functools import partial
import cv2
import sys
import numpy as np
import time

from typing import Dict, List, Tuple
from pathlib import Path

from skimage.filters.rank import median

from abyss.robotics.imaging.image import Image
from abyss.robotics.imaging.processing import process_raw
from abyss.robotics.time.conversions import timestamp_to_name


def dead_pixel_test():
    path = "/home/mga/data/datasets/transect-3/0/les_05/camera_1/20221213T185614.240400000.ppm"
    bayer_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    print("read image")

    blue = bayer_image[1::2, 0::2]  # Blue channel (B)
    green1 = bayer_image[0::2, 1::2]  # First green channel (G1)
    green2 = bayer_image[1::2, 1::2]  # Second green channel (G2)
    red = bayer_image[0::2, 0::2]

    print("separated image")

    filtered_blue = cv2.medianBlur(blue, ksize=3)
    filtered_green_1 = cv2.medianBlur(green1, ksize=3)
    filtered_green_2 = cv2.medianBlur(green2, ksize=3)
    filtered_red = cv2.medianBlur(red, ksize=3)

    print("filtered image")

    bayer_image_out = np.zeros_like(bayer_image)

    bayer_image_out[1::2, 0::2] = filtered_blue  # Blue channel (B)
    bayer_image_out[0::2, 1::2] = filtered_green_1  # First green channel (G1)
    bayer_image_out[1::2, 1::2] = filtered_green_2  # Second green channel (G2)
    bayer_image_out[0::2, 0::2] = filtered_red

    print("combined image")

    # cv2.imwrite("/home/mga/src/local/bayer-tools/image.tiff", bayer_image)

    ts = time.time()
    image = Image(
        camera="camera_1",
        name=timestamp_to_name(ts),
        timestamp=ts,
        image_type="ppm",
        cv_data=bayer_image_out,
    )

    print("processed image")

    image_processed = process_raw(
        raw_image=image,
        debayer_code=48,
        white_balance_values=[2.556, 1.0, 2.1457],
        brightness=6.0,
    )

    image_processed.write(write_path="/home/mga/src/local/bayer-tools/image.png")

    print("wrote image")


def main() -> int:
    """
    Main CLI routine

    Args:
        args: Parsed command line arguments

    Returns:
        Exit status
    """

    print(find_bad_pixel_candidates_bayer2x2())


if __name__ == "__main__":
    sys.exit(main())
