# import json
import time

import fire
from tqdm import tqdm

# from transformers import pipeline
from .Processor import Processor, ProcessorError
from .retrievers.BM25S import RetrieverError


class CLI:

    @staticmethod
    def index(max_chunk_size: int = 2000, bonus: bool = False) -> None:
        processor = Processor(bonus=bonus)
        try:
            processor.index(max_chunk_size)
        except RetrieverError as e:
            raise RetrieverError(e)

    @staticmethod
    def search(query: str, k: int = 5) -> None:
        for i in tqdm(range(1000), desc="Ceci est un loooong test"):
            time.sleep(0.01)

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str,
                       bonus: bool = False) -> None:
        processor = Processor(bonus=bonus)
        try:
            processor.search_dataset(dataset_path, k, save_directory)
        except ProcessorError as e:
            raise ProcessorError("[ERROR]: Search failed") from e

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        pass

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        processor = Processor()
        try:
            processor.answer_dataset(student_search_results_path,
                                     save_directory)
        except ProcessorError as e:
            raise ProcessorError("[ERROR]: Answering failed") from e

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        processor = Processor()
        try:
            processor.evaluate(student_search_results_path, dataset_path)
        except ProcessorError as e:
            raise ProcessorError("[ERROR]: Evaluation failed") from e


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print('\033[H\033[J')
        print("\033[0;32mAborted - See you soon :D\033[0;0m")
    except RetrieverError:
        pass
