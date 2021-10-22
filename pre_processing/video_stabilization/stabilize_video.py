import argparse
import logging
import math
import os
import pathlib
import subprocess
import time
from typing import Dict

import natsort as natsort
import numpy as np
import pandas as pd
import vidstab


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from directory containing videos.")
    parser.add_argument("--root", type=pathlib.Path, help="directory containing videos to be processed",
                        default=pathlib.Path("../Data/test-data/"))
    parser.add_argument("--ext", type=str, help="extensions of the videos within the directory to be processed",
                        default="mp4")
    parser.add_argument("--sh", type=pathlib.Path,
                        help="spreadsheet containing ground truth data of videos to be processed",
                        default=pathlib.Path(__file__).parent / "DOT Iowa Accident Labels.ods")

    return parser.parse_args()


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


def stabilize_vid(dest_dir: pathlib.Path, video_path: pathlib.Path, root: pathlib.PurePath) -> bool:
    """
    :param dest_dir: Directory where the processed frames will be stored.
    :param video_path: The absolute path containing the frames of a video.
    :param root: Directory where the extracted frames are stored.
    :return: True if success
    """
    stabilizer = vidstab.VidStab()
    output_dir = dest_dir/video_path.relative_to(root).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stabilizer.stabilize(os.fspath(video_path), os.fspath(dest_dir/video_path.relative_to(root)))

    return True


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(levelname)s - %(module)s - %(funcName)s - %(message)s")
    hdlr = logging.StreamHandler()
    hdlr.setLevel(logging.DEBUG)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    start_time = time.perf_counter()

    args = parse_args()

    root = args.root.resolve()
    ext = args.ext.split(" ")
    time_f = args.freq
    spreadsheet = args.sh

    repo_path = pathlib.Path(__file__).resolve().parent
    dest_dir = repo_path / "ori_images"
    dest_dir_processed = repo_path / "processed_images"
    video_names = frozenset.union(*frozenset(map(lambda e: frozenset(root.rglob("*." + e)), ext)))
    video_names = natsort.natsorted(video_names, alg=natsort.ns.PATH)
    dfs = pd.read_excel(spreadsheet, sheet_name=None)
    drop_columns(dfs)

    chosen_videos = []
    for df in dfs.values():
        chosen_videos.append(frozenset(filter(lambda v: np.any(df["Video Name"] == v.name), video_names)))
    video_names = frozenset.union(*chosen_videos)
    try:
        subprocess.check_output("nvidia-smi")
        # ctx = decord.gpu(0)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("NVIDIA GPU device not found.")
        # ctx = decord.cpu(0)


    end_time = time.perf_counter()
    res = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Finished in {res}")
