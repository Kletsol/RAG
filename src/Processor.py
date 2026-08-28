from .LoaderSplitter import LoaderSplitter
from .Models import MinimalSearchResults, StudentSearchResults
from .retrievers.BM25S import BM25SRetriever, RetrieverError


class ProcessorError(Exception):
    pass


class Processor:

    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 200:
            raise ProcessorError('[ERROR]: max_chunk_size cannot be lower than 200')
        loader = LoaderSplitter()
        documents = loader.load(max_chunk_size, overlap=50)
        retriever = BM25SRetriever()
        retriever.index(documents, path='./data/tests')
        # try:
        #     retriever.save()
        # except Exception as e:
        #     raise ProcessorError(e)

    def search(query: str, k: int = 5) -> MinimalSearchResults:
        if query == "":
            raise ProcessorError('[ERROR]: Empty query')
        retriever = BM25SRetriever()
        try:
            retriever.load()
        except RetrieverError:
            raise ProcessorError("[ERROR]: Loading failed. No index found.")

    def search_dataset(dataset_path: str, k: int, save_directory: str) -> StudentSearchResults:
        pass

    def answer(query: str, k: int = 5) -> str:
        pass

    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> str:
        pass

    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        pass
