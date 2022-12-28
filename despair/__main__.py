import argparse
import logging
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', type=str, default='warning',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='set the effective log level (debug, info, warning or error)')
    parser.add_argument('-p', '--plot', type=str, choices=['coeff', 'response'],
                        help='plot the given function (coeff, response)')
    parser.add_argument('-r', '--radius', type=int, default=7,
                        help='set the phase filter radius (>0)')

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

    parser.print_usage()
    sys.exit(1)


if __name__ == '__main__':
    setup_logging()

    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler: '{e}'")
        sys.exit(1)
