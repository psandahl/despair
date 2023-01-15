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


def main() -> bool:
    """
    Entry point for application.
    """
    parser = argparse.ArgumentParser(prog='python -m despair')
    parser.add_argument('--log', type=str, default='warning',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='set the effective log level (default: warning)')
    parser.add_argument('--plot', type=str, choices=['coeff', 'response', 'shift', 'disparity'],
                        help='plot the given function')
    parser.add_argument('--disparity-mode', type=str, choices=['feature-image', 'single-gt', 'pair', 'multi'],
                        help='run disparity in the given mode')
    parser.add_argument('--shift-mode', type=str, choices=['global', 'peak'],
                        help='shift mode for disparity ground truth')
    parser.add_argument('--shift-scale', type=float, default=1.0,
                        help='set the scale for the ground thruth shift (default: 1)')
    parser.add_argument('--reference', type=pathlib.Path,
                        help='the reference image')
    parser.add_argument('--query', type=pathlib.Path,
                        help='the query image')
    parser.add_argument('--radius', type=int, choices=range(1, 10), default=7,
                        help='set the phase filter radius (default: 7)')
    parser.add_argument('--max-levels', type=int, choices=range(-1, 8), default=3,
                        help='set the maximum scale level (default: 3)')
    parser.add_argument('--target-level', type=int, choices=range(0, 8), default=0,
                        help='set the target scale level (default: 0)')

    args = parser.parse_args()

    # Adjust the effective log level.
    log_level = args.log.upper()
    handler.setLevel(getattr(logging, log_level))

    # Check for plotting.
    if args.plot == 'coeff':
        plot.coeff(args.radius)
        return True
    elif args.plot == 'response':
        if not args.reference is None:
            return plot.response_image(args.reference, args.radius, args.target_level)
        else:
            plot.response_feature_image(args.radius)
            return True
    elif args.plot == 'shift':
        if not args.reference is None and not args.shift_mode is None:
            return plot.shift(
                args.reference, args.shift_mode, args.shift_scale)
        else:
            parser.print_usage()
            return False
    elif args.plot == 'disparity':
        if args.disparity_mode == 'feature-image':
            plot.disparity_feature_image(args.radius, args.shift_scale)
            return True
        elif args.disparity_mode == 'single-gt':
            if not args.reference is None and not args.shift_mode is None:
                return plot.disparity_single(
                    args.reference, args.shift_mode, args.shift_scale, args.radius, args.target_level)
            else:
                parser.print_usage()
                return False
        elif args.disparity_mode == 'pair':
            if not args.reference is None and not args.query is None:
                return plot.disparity_pair(
                    args.reference, args.query, args.radius, args.target_level)
            else:
                parser.print_usage()
                return False
        elif args.disparity_mode == 'multi':
            if not args.reference is None and not args.query is None:
                return plot.disparity_multi(
                    args.reference, args.query, args.radius)
            else:
                parser.print_usage()
                return False
        else:
            parser.print_usage()
            return False

    parser.print_usage()
    return False


if __name__ == '__main__':
    setup_logging()

    try:
        sys.exit(0 if main() == True else 1)
    except Exception as e:
        logger.exception(f"Global exception handler: '{e}'")
        sys.exit(1)
