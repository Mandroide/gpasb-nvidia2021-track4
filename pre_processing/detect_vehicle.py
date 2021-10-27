#!/usr/bin/env python
import logging
import subprocess
import time


# Following the common practice, we apply the Feature Pyramid Network (FPN) [18] to build high-level semantic feature
# maps at all scales. Figure 5 shows the results of our detector
def train_faster_rcnn(sgd_with_momentum: float = 0.9, weight_decay: float = 1e-4, learning_rate: float = 0.01,
                      minibatch: int = 8, n_iterations: int = 50_000) -> None:
    """Main detect for vehicle detection with SENet-152.

    :return:
    """
    pass


def train_cascade_rcnn() -> None:
    """Auxiliary detect for reducing failing of vehicle detection with CBResnet-200.

    :return:
    """
    pass


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
    try:
        subprocess.check_output("nvidia-smi")
        # Install PaddleDetection GPU
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("NVIDIA GPU device not found.")
        # Install PaddleDetection CPU

    end_time = time.perf_counter()
    res = time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time))
    print(f"Finished in {res}")
    return 0


if __name__ == "__main__":
    main()
