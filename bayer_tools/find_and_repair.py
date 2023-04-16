import time
import cv2
import numpy as np
import tqdm

from typing import List
from pathlib import Path

from abyss.bedrock.typing import GenericPath
from abyss.robotics.imaging.image import Image
from abyss.robotics.imaging.processing import process_raw
from abyss.robotics.utils.sampling import uniform_sample_count


def find_bad_pixels(
    image_paths: List[GenericPath],
    image_count: int = 50,
    confirm_ratio: float = 0.9,
    kernal_size: int = 3,
) -> np.ndarray:
    """Find and return coordinates of hot/dead pixels in the given RAW images.

    The probability that a detected bad pixel is really a bad pixel gets
    higher the more input images are given. The images should be taken around
    the same time, that is, each image must contain the same bad pixels.
    Also, there should be movement between the images to avoid the false detection
    of bad pixels in non-moving high-contrast areas.

    Args:
        image_paths (List[GenericPath]): paths to raw images should be .ppm
        image_count (int, optional): how many images to use, if <= 0 will use all images. Defaults to 50.
        confirm_ratio (float, optional): ratio of how many out of all given images
        must contain a bad pixel to confirm it as such. Defaults to 0.9.
        kernal_size (int): kernal size in pixels used for medianBlur

    Returns:
        np.ndarray: coordinates of confirmed bad pixels,
        ndarray of shape (n,2) with y,x coordinates of raw image
    """

    pixel_coords = []

    if image_count >= 0 and image_count < len(image_paths):
        image_paths = uniform_sample_count(input_list=image_paths, count=image_count)

    width = None

    for image_path in tqdm.tqdm(image_paths):
        raw_image = Image.load_image(
            path=image_path,
            camera="",
            load_data=True,
            timestamp=time.time(),
        )
        # we need the width later for counting
        width = raw_image.width
        # threshold used for computing candidates
        thresh = max(np.max(raw_image.cv_data) // 150, 20)

        pixel_coords.extend(
            _find_bad_pixel_candidates(
                raw_image.cv_data, thresh=thresh, kernal_size=kernal_size
            )
        )

    pixel_coords = np.vstack(pixel_coords)

    if len(image_paths) == 1:
        return pixel_coords

    # select candidates that appear on most input images
    # count how many times a coordinate appears

    # first we convert y,x to array offset such that we have an array of integers
    offset = pixel_coords[:, 0] * width
    offset += pixel_coords[:, 1]

    # now we count how many times each offset occurs, see: https://stackoverflow.com/a/4652265
    offset.sort()
    diff = np.concatenate(([1], np.diff(offset)))
    idx = np.concatenate((np.where(diff)[0], [len(offset)]))
    # note: the following is faster than np.transpose([vals,cnt])
    vals = offset[idx[:-1]]
    cnt = np.diff(idx)
    res = np.empty((len(vals), 2), dtype=vals.dtype)
    res[:, 0] = vals
    res[:, 1] = cnt
    counts = res

    # we select the ones whose count is high
    is_bad = counts[:, 1] >= confirm_ratio * len(image_paths)

    # and convert back to y,x
    bad_offsets = counts[is_bad, 0]
    bad_coords = np.transpose([bad_offsets // width, bad_offsets % width])

    return bad_coords


def _find_bad_pixel_candidates(
    image_data: np.ndarray, thresh: int, kernal_size: int = 3
) -> List[np.ndarray]:
    """Finds bad pixel canditates for each bayer color space

    create a view for each color, do 3x3 median on it, find bad pixels, correct coordinates
    This shortcut allows to do median filtering without using a mask, which means
    that OpenCV's extremely fast median filter algorithm can be used.

    Args:
        image_data (np.ndarray): input image
        thresh (int): threshold to determine bad pixel
        kernal_size (int): kernal size in pixels used for medianBlur

    Returns:
        List[np.ndarray]: array of bad pixel candidates for each color space
    """

    coords = []

    # we have 4 colors (two greens are always seen as two colors)
    for offset_y in [0, 1]:
        for offset_x in [0, 1]:
            image_slice = image_data[offset_y::2, offset_x::2]

            image_slice = np.require(image_slice, image_slice.dtype, "C")
            median = cv2.medianBlur(image_slice, ksize=kernal_size)

            # detect possible bad pixels
            np.subtract(image_slice, median, out=median)
            np.abs(median, out=median)
            candidates = median > thresh

            # convert to coordinates and correct for slicing
            y, x = np.nonzero(candidates)
            # note: the following is much faster than np.transpose((y,x))
            candidates = np.empty((len(y), 2), dtype=y.dtype)
            candidates[:, 0] = y
            candidates[:, 1] = x

            candidates *= 2
            candidates[:, 0] += offset_y
            candidates[:, 1] += offset_x

            coords.append(candidates)

    return coords


def repair_bad_pixels(image: Image, coords: np.ndarray, kernal_size: int = 3) -> None:
    """Repair bad pixels in the given raw image.

    Args:
        image (Image): image to repair
        coords (np.ndarray): coordinates of confirmed bad pixels,
        ndarray of shape (n,2) with y,x coordinates of raw image
        kernal_size (int): kernal size in pixels used for medianBlur
    """
    image_data = image.cv_data

    # we have 4 colors (two greens are always seen as two colors)
    for offset_y in [0, 1]:
        for offset_x in [0, 1]:
            image_slice = image_data[offset_y::2, offset_x::2]

            image_slice_cv = np.require(image_slice, image_slice.dtype, "C")
            smooth = cv2.medianBlur(image_slice_cv, ksize=kernal_size)

            # determine which bad pixels belong to this color slice
            sliced_y = coords[:, 0] - offset_y
            sliced_y %= 2
            sliced_x = coords[:, 1] - offset_x
            sliced_x %= 2
            matches_slice = sliced_y == 0
            matches_slice &= sliced_x == 0

            coords_color = coords[matches_slice]

            # convert the full-size coordinates to the color slice coordinates
            coords_color[:, 0] -= offset_y
            coords_color[:, 1] -= offset_x
            coords_color //= 2

            mask = np.zeros_like(image_slice, dtype=bool)
            mask[coords_color[:, 0], coords_color[:, 1]] = True

            image_slice[mask] = smooth[mask]

    return image


if __name__ == "__main__":
    image_paths = [
        str(path)
        for path in sorted(
            Path("/home/mga/data/datasets/transect-3/0/les_05/camera_1").iterdir()
        )
    ]

    coords = find_bad_pixels(image_paths, confirm_ratio=0.9)

    print(f"found {len(coords)} bad pixels")

    output_dir = Path("/home/mga/src/local/bayer-tools/output")
    output_dir.mkdir(exist_ok=True)

    for image_path in tqdm.tqdm(image_paths):
        raw_image = Image.load_image(
            path=image_path,
            camera="camera_1",
            encoding="bayer_bggr16",
            load_data=True,
            timestamp=time.time(),
        )

        image = repair_bad_pixels(raw_image, coords=coords, kernal_size=3)

        image_processed = process_raw(
            raw_image=image,
            debayer_code=48,
            white_balance_values=[2.556, 1.0, 2.1457],
            brightness=6.0,
        )

        image_processed.write(write_directory=str(output_dir))
