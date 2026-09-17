import fire

from .Evaluator import EvaluationError, Evaluator
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
    def search(query: str, k: int = 5, bonus: bool = False) -> None:
        processor = Processor(bonus=bonus)
        try:
            result = processor.search(query, k)
        except ProcessorError as e:
            raise ProcessorError(e)
        for row in result.retrieved_sources:
            print(
                f"{row.file_path} "
                f"[{row.first_character_index}:{row.last_character_index}]"
            )

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str,
                       bonus: bool = False) -> None:
        processor = Processor(bonus=bonus)
        try:
            processor.search_dataset(dataset_path, k, save_directory)
        except ProcessorError as e:
            raise ProcessorError(e)

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        processor = Processor()
        try:
            answer = processor.answer(query, k)
        except ProcessorError as e:
            raise ProcessorError(e)
        print(answer)

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        processor = Processor()
        try:
            processor.answer_dataset(student_search_results_path,
                                     save_directory)
        except ProcessorError as e:
            raise ProcessorError(f"Answering failed: {e}")

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        evaluator = Evaluator()
        try:
            results = evaluator.evaluate(student_search_results_path,
                                         dataset_path)
        except EvaluationError as e:
            raise EvaluationError(e)
        print(f"--- Evaluation results ---\n"
              f"Recall@1: {int(results.recall1 * 100)}%\n"
              f"Recall@3: {int(results.recall3 * 100)}%\n"
              f"Recall@5: {int(results.recall5 * 100)}%\n"
              f"Recall@10: {int(results.recall10 * 100)}%\n")


if __name__ == "__main__":
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print('\033[H\033[J')
        print("\033[0;32mAborted - See you soon :D\033[0;0m")
    except (RetrieverError, ProcessorError, EvaluationError) as e:
        print(e)
