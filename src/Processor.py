from multiprocessing import Process

from .Retriever import BM25SRetriever

class ProcessorError(Exception):
    pass


class Processor:

    def index(max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 200:
            raise ProcessorError('[ERROR]: max_chunk_size cannot be lower than 200')
        retriever = BM25SRetriever()
        retriever.index(max_chunk_size, overlap=15/100)
        try:
            retriever.export()
        except Exception:
            raise ProcessorError('test')

    def search(query: str, k: int = 5) -> None:
        if query == "":
            raise ProcessorError('[ERROR]: Empty query')

    def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
        pass

    def answer(query: str, k: int = 5) -> None:
        pass

    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        pass

    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        pass
