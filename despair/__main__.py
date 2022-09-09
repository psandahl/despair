import argparse
import logging
import sys

import cv2 as cv
import matplotlib.pyplot as plt

logger = None
handler = None


def setup_logging() -> None:
    """
    Global setup of logging system. Module loggers then register
    as getLogger(__name__) to end up in logger tree.
    """
    global logger
    logger = logging.getLogger('dapper')
    logger.setLevel(logging.DEBUG)

    global handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '[%(levelname)s %(name)s:%(lineno)d] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def main() -> None:
    """
    Entry point for application.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',
                        help='set the effective log level (DEBUG, INFO, WARNING or ERROR)')
    parser.add_argument('-l', '--left',
                        help='path to the left image')
    parser.add_argument('-r', '--right',
                        help='path to the right image')
    parser.add_argument('-g', '--ground-truth',
                        help='path to ground truth image')
    args = parser.parse_args()

    # Check if the effective log level shall be altered.
    if not args.log is None:
        log_level = args.log.upper()
        if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            num_log_level = getattr(logging, log_level)
            handler.setLevel(num_log_level)
        else:
            parser.print_help()
            sys.exit(1)

    left = None
    right = None
    ground_truth = None

    if not args.left is None:
        left = cv.imread(args.left, cv.IMREAD_GRAYSCALE)
        if left is None:
            logger.error('Failed to read left image')
            sys.exit(1)

    if not args.right is None:
        right = cv.imread(args.right, cv.IMREAD_GRAYSCALE)
        if right is None:
            logger.error('Failed to read right image')
            sys.exit(1)

    if not args.ground_truth is None:
        ground_truth = cv.imread(
            args.ground_truth, cv.IMREAD_GRAYSCALE | cv.IMREAD_UNCHANGED)
        if ground_truth is None:
            logger.error('Failed to read ground truth image')
            sys.exit(1)

    if left is None or right is None:
        parser.print_help()
        sys.exit(1)

    plt.subplot(211)
    plt.title('Left image')
    plt.imshow(left, cmap='gray', vmin=0, vmax=255)

    plt.subplot(212)
    plt.title('Right image')
    plt.imshow(right, cmap='gray', vmin=0, vmax=255)

    plt.show()

    # Successful exit.
    sys.exit(0)


if __name__ == '__main__':
    setup_logging()

    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler: '{e}'")
        sys.exit(1)
