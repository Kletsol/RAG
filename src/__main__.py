import fire
import time
from tqdm import tqdm
from .Processor import Processor
from .Retriever import RetrieverError


class CLI:

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        processor = Processor()
        try:
            processor.index(max_chunk_size=max_chunk_size)
        except RetrieverError as e:
            raise RetrieverError(e)

    @staticmethod
    def search(query: str, k: int = 5) -> None:
        for i in tqdm(range(1000), desc="Ceci est un loooong test"):
            time.sleep(0.01)

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
        pass

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        pass

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        pass

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        pass


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print('\033[H\033[J')
        print("\033[0;32mAborted - See you soon :D\033[0;0m")
    except RetrieverError:
        pass
