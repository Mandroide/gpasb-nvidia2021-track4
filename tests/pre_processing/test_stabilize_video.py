import os
import pathlib
import sys
import pytest

sys.path.append(os.fspath(pathlib.PurePath(__file__).parent.parent.parent))

from pre_processing import stabilize_video


@pytest.mark.parametrize("dest_dir, video_path, root")
def test_stabilization(dest_dir: pathlib.Path, video_path: pathlib.Path, root: pathlib.PurePath):
    res = None  # stabilize_video.process_frames(None, None, None)
    assert res
