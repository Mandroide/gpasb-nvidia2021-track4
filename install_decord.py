#!/usr/bin/env python
import os
import pathlib
import subprocess
import sys


def install_decord(repo_path: pathlib.Path) -> None:
    """
    :param repo_path: Directory where it will be installed.
    :return:
    """
    sudo = subprocess.Popen(["sudo", "add-apt-repository", "-y", "ppa:jonathonf/ffmpeg-4"])
    sudo.wait()
    sudo = subprocess.Popen(["sudo", "apt-get", "update"])
    sudo.wait()
    sudo = subprocess.Popen(
        ["sudo", "apt-get", "install", "-y", "build-essential", "python3-dev", "python3-setuptools",
         "make", "cmake", "ffmpeg", "libavcodec-dev", "libavfilter-dev", "libavformat-dev",
         "libavutil-dev"])
    os.chdir(repo_path)
    git = subprocess.Popen(
        ["git", "clone", "--depth", "1", "--recursive", "https://github.com/dmlc/decord"])
    git.wait()
    build_dir = repo_path / "decord/build"
    build_dir.mkdir(exist_ok=True)
    os.chdir(build_dir)
    sudo.wait()
    cmake = subprocess.Popen(["cmake", "..", "-DUSE_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"])
    cmake.wait()
    cmake = subprocess.Popen("make")
    cmake.wait()
    os.chdir("../python")
    os.system('python3 setup.py install --user')
    sys.path.append(os.fspath(repo_path / "decord/python"))


if __name__ == "__main__":
    repo_path = pathlib.Path(__file__).resolve().parent
    install_decord(repo_path)
