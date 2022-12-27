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
    parser.add_argument('-l', '--log', type=str, default='warning',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='set the effective log level (debug, info, warning or error)')
    parser.add_argument('-c', '--plot-coeff', type=int,
                        help='plot the filter coeffients using the given radius')
    parser.add_argument('-r', '--plot-response', type=int,
                        help='plot the filter response using the given radius')

    args = parser.parse_args()

    # Adjust the effective log level.
    log_level = args.log.upper()
    handler.setLevel(getattr(logging, log_level))

    # Check arguments.
    if args.plot_coeff and args.plot_coeff > 0:
        plot.coeff(args.plot_coeff)
    elif args.plot_response and args.plot_response > 0:
        plot.response(args.plot_response)
    else:
        parser.print_usage()


if __name__ == '__main__':
    setup_logging()

    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler: '{e}'")
        sys.exit(1)
