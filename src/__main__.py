import fire


class CLI:

    def index(max_chunk_size: int = 2000) -> None:
        pass

    def search(query: str, k: int = 5) -> None:
        pass

    def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
        pass

    def answer(query: str, k: int = 5) -> None:
        pass

    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        pass

    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        pass


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print('\033[H\033[J')
        print("\033[0;32mAborted - See you soon :D\033[0;0m")
