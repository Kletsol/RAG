import json
import os

import fire

from .Evaluator import Evaluator, EvaluatorError
from .Processor import Processor, ProcessorError
from .retrievers.BM25S import RetrieverError


class CLI:

    @staticmethod
    def index(max_chunk_size: int = 2000, bonus: bool = False) -> None:
        """Indexes a whole batch of files

        Args:
            max_chunk_size (int, optional): The max size possible for
                                            every chunk in a splitted file.
            bonus (bool, optional): Set to true for hybrid RAG.
                                    Defaults to False.
        """
        processor = Processor(activate_bonus=bonus)
        try:
            processor.index(max_chunk_size)
        except RetrieverError as e:
            raise RetrieverError(e)
        print("\033[1;34m[SUCCESS] - Corpus indexed and saved\033[0;0m")

    @staticmethod
    def search(query: str, k: int = 5, bonus: bool = False) -> None:
        """Searches the top-k relevant sources for a given query

        Args:
            query (str): The query to process
            k (int, optional): The number of sources returned. Defaults to 5.
            bonus (bool, optional): Set to True for hybrid RAG.
                                    Defaults to False.
        """
        processor = Processor(activate_bonus=bonus)
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
        """Searches the top-k relevant sources for each question in the
           dataset and saves it

        Args:
            dataset_path (str): The dataset to process
            k (int): The number of sources returned.
            save_directory (str): The folder in which to save the result
            bonus (bool, optional): Set to True for hybrid RAG.
                                    Defaults to False.
        """
        processor = Processor(activate_bonus=bonus)
        try:
            student_results = processor.search_dataset(dataset_path, k)
        except ProcessorError as e:
            raise ProcessorError(e)
        # ------
        # Save
        # ------
        file_basename = os.path.basename(dataset_path)
        try:
            os.makedirs(save_directory, exist_ok=True)
            output_path = (f"{save_directory}/{file_basename}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(student_results.model_dump(), f,
                          ensure_ascii=False, indent=2)
        except OSError as e:
            raise ProcessorError(f"[ERROR]: Cannot save search results - {e}")
        print("\033[1;34m[SUCCESS] - Search results saved\033[0;0m")

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        """Tries answering the query using the top-k most relevant sources

        Args:
            query (str): The query to process
            k (int, optional): The number of sources to use. Defaults to 5.
        """
        processor = Processor()
        try:
            answer = processor.answer(query, k)
        except ProcessorError as e:
            raise ProcessorError(e)
        print(answer)

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        """Generates answers for a previously generated dataset,
        producing a StudentSearchResultsAndAnswer JSON file

        Args:
            student_search_results_path (str): The dataset to process
            save_directory (str): The folder in which to save the results
        """
        processor = Processor()
        try:
            final_results = processor.answer_dataset(
                student_search_results_path)
        except ProcessorError as e:
            raise ProcessorError(f"Answering failed: {e}")
        # ------
        # Save
        # ------
        file_basename = os.path.basename(student_search_results_path)
        try:
            os.makedirs(save_directory, exist_ok=True)
            output_path = (f"{save_directory}/{file_basename}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_results.model_dump(), f, ensure_ascii=False,
                          indent=2)
        except (OSError, PermissionError) as e:
            raise ProcessorError(f"[ERROR]: Could not save answers - {e}")
        print("\033[1;34m[SUCCESS] - Queries answered and saved\033[0;0m")

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        """Reports the student's recall against a ground_truth dataset

        Args:
            student_search_results_path (str): The student's dataset path
            dataset_path (str): The ground_truth dataset path
        """
        evaluator = Evaluator()
        try:
            results = evaluator.evaluate(student_search_results_path,
                                         dataset_path)
        except EvaluatorError as e:
            raise EvaluatorError(e)
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
    except (RetrieverError, ProcessorError, EvaluatorError) as e:
        print(f"\033[0;31m{e}\033[0;0m")
