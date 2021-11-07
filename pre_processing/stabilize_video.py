#!/usr/bin/env python
import argparse
import concurrent.futures
import json
import logging
import math
import os
import pathlib
import subprocess
import time
from typing import TextIO, Dict

import cv2
import vidstab

import decord
import natsort
import numpy as np
import pandas as pd
import tqdm


def extract_frames(dest_dir: pathlib.Path, video_path: pathlib.PurePath, root: pathlib.PurePath, time_f: int,
                   ori_images_txt: TextIO, ctx) -> pathlib.Path:
    """
    :param dest_dir: Directory where the extracted frames will be stored.
    :param video_path: The absolute path of the video.
    :param root: Directory containing the videos to be processed.
    :param time_f: Time frequency.
    :param ori_images_txt: File object where the frames of `video_path` are stored.
    :param ctx: The context to decode the video file.
    :return: Directory containing the stored frames of `video_path`
    """
    pic_path = dest_dir / video_path.relative_to(root).with_suffix("")
    pic_path.mkdir(parents=True, exist_ok=True)
    try:
        vr = decord.VideoReader(os.fspath(video_path), ctx=ctx)
        vr.skip_frames(time_f)
        vr.seek(time_f - 1)
        size = len(vr)
        frames_indices = range(time_f - 1, size, time_f)
        for c in frames_indices:
            img_path = os.fspath(pic_path / (str(c + 1) + '.jpg'))
            cv2.imwrite(img_path, vr.next().asnumpy())
            ori_images_txt.write(img_path + "\n")
    except decord.DECORDError:
        vc = cv2.VideoCapture(os.fspath(video_path))
        if vc.isOpened():
            c = 1
            while vc.grab():
                if c % time_f == 0:
                    img_path = os.fspath(pic_path / (str(c) + '.jpg'))
                    _, frame = vc.retrieve()
                    cv2.imwrite(img_path, frame)
                    ori_images_txt.write(img_path + "\n")
                c += 1
                cv2.waitKey(1)
            vc.release()
    return pic_path


def stabilize_frames(dest_dir: pathlib.Path) -> bool:
    """
    :param dest_dir: Directory where the processed frames are stored.
    :return: True if success
    """
    files = natsort.natsorted(dest_dir.glob("*.jpg"), alg=natsort.ns.PATH)
    stabilizer = vidstab.VidStab()
    success = False
    for f in files:
        frame = cv2.imread(os.fspath(f))
        frame = stabilizer.stabilize_frame(frame)
        cv2.imwrite(os.fspath(f), frame)
        success = True
    return success


def stabilize_video(stabilizer: vidstab.VidStab, dest_dir: pathlib.Path, video_path: pathlib.PurePath,
                    root: pathlib.PurePath) -> pathlib.Path:
    """

    :param stabilizer: VidStab object with stabilize method.
    :param dest_dir: Directory where the stabilized video will be stored.
    :param video_path: Video to stabilize
    :param root: Directory with the source video to stabilize.
    :return: Video path of the stabilized video
    """
    logger = get_logger()
    vid_path = dest_dir / video_path.relative_to(root)
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Printing the vid_path {os.fspath(vid_path.with_suffix('avi'))}")
    stabilizer.stabilize(input_path=os.fspath(video_path), output_path=os.fspath(vid_path.with_suffix("mp4")))
    return vid_path.with_suffix("mp4")


def parse_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frames from directory containing videos.")
    parser.add_argument("--root", type=pathlib.Path, help="directory containing videos to be processed",
                        default=pathlib.Path(__file__).parent / "sample_video")
    parser.add_argument("--ext", type=str, help="extensions of the videos within the directory to be processed",
                        default="mp4")
    parser.add_argument("--freq", type=int, help="time frequency", default=10)
    parser.add_argument("--sh", type=pathlib.Path,
                        help="spreadsheet containing ground truth data of videos to be processed",
                        default=None)

    return parser


def drop_columns(dfs: Dict) -> None:
    for year in dfs.keys():
        # Replace empty values with NaN
        dfs[year].replace("", math.nan, inplace=True)
        dfs[year].replace("NA", math.nan, inplace=True)

        # Drop NaN values.
        columns = dfs[year].columns
        dfs[year].columns = pd.RangeIndex(0, len(columns))
        dfs[year].dropna(subset=[7, 8, 9], inplace=True)

        # Drop unnecessary columns
        dfs[year].columns = columns
        dfs[year].drop(columns=list(dfs[year].columns[0:3]) + [dfs[year].columns[5]]
                               + list(dfs[year].columns[10:]), inplace=True)
        dfs[year] = dfs[year].astype(int, errors="ignore")


def get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(levelname)s - %(module)s - %(funcName)s - %(message)s")
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)

    return logger


def main(args: argparse.Namespace) -> int:
    logger = get_logger()
    start_time = time.perf_counter()

    root = args.root.resolve()
    ext = args.ext.split(" ")
    time_f = args.freq
    spreadsheet = args.sh

    repo_path = pathlib.Path(__file__).resolve().parent.parent
    dest_dir = repo_path / "processed_images"
    video_names = frozenset.union(*frozenset(map(lambda e: frozenset(root.rglob("*." + e)), ext)))
    video_names = natsort.natsorted(video_names, alg=natsort.ns.PATH)

    if spreadsheet:
        dfs = pd.read_excel(spreadsheet, sheet_name=None)
        drop_columns(dfs)
        chosen_videos = []
        for df in dfs.values():
            chosen_videos.append(frozenset(filter(lambda v: np.any(df["Video Name"] == v.name), video_names)))
        video_names = frozenset.union(*chosen_videos)

    try:
        subprocess.check_output("nvidia-smi")
        ctx = decord.gpu(0)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("NVIDIA GPU device not found.")
        ctx = decord.cpu(0)

    with open(repo_path / "ori_images.txt", "w") as ori_images_txt, \
            concurrent.futures.ProcessPoolExecutor() as p_exec, concurrent.futures.ThreadPoolExecutor() as t_exec:
        fs = [t_exec.submit(extract_frames, dest_dir, video_path, root, time_f, ori_images_txt, ctx)
              for video_path in video_names]
        for future in tqdm.tqdm(concurrent.futures.as_completed(fs)):
            p_exec.submit(stabilize_frames, future.result())
        stabilizer = vidstab.VidStab()
        dest_dir = repo_path / "stabilized_videos"
        fs = [t_exec.submit(stabilize_video, stabilizer, dest_dir, video_path, root)
              for video_path in video_names]
        for future in tqdm.tqdm(concurrent.futures.as_completed(fs)):
            print(os.fspath(future.result()))

    # Store relative paths to root directory.
    with open(repo_path / "dataset.json", "w") as f:
        relative_paths = list(map(lambda v: os.fspath(v.relative_to(root)), video_names))
        json.dump(relative_paths, f)

    end_time = time.perf_counter()
    res = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Finished in {res}")

    return 0


if __name__ == "__main__":
    main(parse_cli().parse_args())
