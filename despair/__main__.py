import argparse
import logging
import pathlib
import sys

import despair.plot as plot

logger = None
handler = None


def setup_logging() -> None:
    """
    Global setup of logging system. Module loggers then register
    as getLogger(__name__) to end up in logger tree.
    """
    global logger
    logger = logging.getLogger('despair')
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
    parser = argparse.ArgumentParser(prog='python -m despair')
    parser.add_argument('--log', type=str, default='warning',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='set the effective log level (default: warning)')
    parser.add_argument('--plot', type=str, choices=['coeff', 'response', 'shift'],
                        help='plot the given function')
    parser.add_argument('--disparity-mode', type=str, choices=['ground-truth'],
                        help='run disparity in the given mode')
    parser.add_argument('--shift-mode', type=str, choices=['global'],
                        help='shift mode for ground truth disparity')
    parser.add_argument('--shift-scale', type=float, default=1.0,
                        help='set the scale for the ground thruth shift (default: 1)')
    parser.add_argument('--reference', type=pathlib.Path,
                        help='the reference image')
    parser.add_argument('--radius', type=int, choices=range(1, 10), default=7,
                        help='set the phase filter radius (default: 7)')
    parser.add_argument('--scale-space', type=int, choices=range(0, 10), default=3,
                        help='set the maximum scale space level (default: 3)')

    args = parser.parse_args()

    # Adjust the effective log level.
    log_level = args.log.upper()
    handler.setLevel(getattr(logging, log_level))

    # Check for plotting.
    if args.plot == 'coeff':
        plot.coeff(args.radius)
        sys.exit(0)
    elif args.plot == 'response':
        plot.response(args.radius)
        sys.exit(0)
    elif args.plot == 'shift':
        if not args.reference is None and not args.shift_mode is None:
            result = plot.shift(
                args.reference, args.shift_mode, args.shift_scale)
            sys.exit(0 if result else 1)
        else:
            parser.print_usage()
            sys.exit(1)

    parser.print_usage()
    sys.exit(1)


if __name__ == '__main__':
    setup_logging()

    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler: '{e}'")
        sys.exit(1)
