from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    """
    Parses arguments given at execution using argparse

    Returns:
        Namespace: the parsed arguments
    """
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--max_chunk_size",
                        help="Configurable chunk size",
                        default=2000,
                        required=False)
    parsed = parser.parse_args()
    return parsed
