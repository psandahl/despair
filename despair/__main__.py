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
    parser.add_argument('-f', '--plot-filters', type=int,
                        help='plot the available filters using the given radius')
    parser.add_argument('-r', '--plot-responses', type=int,
                        help='plot the filter responses using the given radius')

    args = parser.parse_args()

    # Adjust the effective log level.
    log_level = args.log.upper()
    handler.setLevel(getattr(logging, log_level))

    # Check arguments.
    if args.plot_filters and args.plot_filters > 0:
        plot.filters(args.plot_filters)
    elif args.plot_responses and args.plot_responses > 0:
        plot.responses(args.plot_responses)
    else:
        parser.print_usage()


if __name__ == '__main__':
    setup_logging()

    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler: '{e}'")
        sys.exit(1)
