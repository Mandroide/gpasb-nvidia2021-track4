#!/usr/bin/env python
import logging
import os
import pathlib
import time

import cv2
import natsort


def model(frames_dir: pathlib.Path) -> None:
    """Model backward and forward background.

    :param frames_dir: Directory with the jpg images
    :return:
    """
    files = natsort.natsorted(frames_dir.glob("*.jpg"), alg=natsort.ns.PATH)
    pre_processing_folder = pathlib.Path(__file__).parent
    file_folder = files[0].relative_to(pre_processing_folder).parent
    save_fw_path_ = pre_processing_folder / "bg/forward" / file_folder
    save_fw_path_.mkdir(parents=True, exist_ok=True)
    save_bw_path_ = pre_processing_folder / "bg/backward" / file_folder
    save_bw_path_.mkdir(parents=True, exist_ok=True)
    bs = cv2.createBackgroundSubtractorMOG2()
    for f in files:
        model_background(f, bs, save_fw_path_, save_bw_path_)


def model_background(filename: pathlib.PurePath, bs: cv2.BackgroundSubtractorMOG2, save_fw_path: pathlib.PurePath,
                     save_bw_path: pathlib.PurePath) -> None:
    """Model background using mixture of Gaussians (MOG).

    :param filename: source image for background modeling.
    :param bs: background substractor for forward and backward background.
    :param save_fw_path: output directory of forward background frame.
    :param save_bw_path: output directory of backward background frame.
    :return:
    """
    frame = cv2.imread(os.fspath(filename))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fw_img = bs.apply(gray)
    bw_img = bs.getBackgroundImage()
    cv2.imwrite(os.fspath(save_bw_path / '{}.png'.format(filename.stem)), bw_img)
    cv2.imwrite(os.fspath(save_fw_path / '{}.png'.format(filename.stem)), fw_img)


def main() -> int:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(levelname)s - %(module)s - %(funcName)s - %(message)s")
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    start_time = time.perf_counter()

    # TODO: Implement module
    frames_dir_ = pathlib.Path("")
    model(frames_dir_)

    end_time = time.perf_counter()
    res = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Finished in {res}")
    return 0


if __name__ == "__main__":
    main()
