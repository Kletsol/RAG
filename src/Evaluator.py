import json

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .Models import AnsweredQuestion, RagDataset, StudentSearchResults


class EvaluationError(Exception):
    """
    Errors related to the evaluation process.
    """
    pass


class EvaluationResult(BaseModel):
    """
    Class representing an evaluation result.
    """
    recall1: float
    recall3: float
    recall5: float
    recall10: float
    questions_evaluated: int


class Evaluator:
    def evaluate(self, student_search_results_path: str, dataset_path: str
                 ) -> EvaluationResult:
        """
        Evaluate the results of the RAG with ground truth
        datas to give a ratio of performances.
        """
        # # Load student results
        # try:
        #     with open(student_search_result_path, 'r') as f:
        #         search_results = (
        #             StudentSearchResults.model_validate(json.load(f)))
        # except (FileNotFoundError, OSError, UnicodeDecodeError):
        #     raise EvaluationError('Unable to read the '
        #                           'search_results properly.')
        # except ValidationError:
        #     raise EvaluationError('Your student_search_result_path '
        #                           'file is corrupted. Cannot read it.')

        # # Load ground-truth dataset
        # try:
        #     with open(dataset_path, 'r') as f:
        #         data = json.load(f)
        #     ground_truth = RagDataset.model_validate(data)
        # except (FileNotFoundError, OSError, UnicodeDecodeError):
        #     raise EvaluationError('Unable to read the '
        #                           'ground truth dataset properly.')
        # except ValidationError:
        #     raise EvaluationError('Your ground truth dataset '
        #                           'file is corrupted. Cannot read it.')

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_results = (
                    StudentSearchResults.model_validate(json.load(f)))
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValueError):
            raise EvaluationError("[ERROR]: Could not load evaluation data")

        # Calculate results
        values = [1, 3, 5, 10]
        recalls = []
        for k in tqdm(values, desc="Calculating recall@k", colour='cyan'):
            recalls.append(self.get_recall(student_results, dataset, k))

        try:
            results = EvaluationResult(recall1=round(recalls[0], 2),
                                       recall3=round(recalls[1], 2),
                                       recall5=round(recalls[2], 2),
                                       recall10=round(recalls[3], 2),
                                       questions_evaluated=len(
                                           student_results.search_results))
        except ValidationError:
            raise EvaluationError('Could not evaluate the resuls.')

        return results

    @staticmethod
    def get_recall(
        student_results: StudentSearchResults,
        dataset: RagDataset,
            k: int) -> float:
        """
        Calculate the recall@k score over the whole dataset
        compared to the ground-truth dataset.

        Return a ratio of good retrieving (overlap of 0.05% at least)
        """
        ground_truth = {}

        for question in dataset.rag_questions:
            if isinstance(question, AnsweredQuestion):
                ground_truth[question.question_id] = question.sources
        recalls = []
        for result in student_results.search_results:
            expected_sources = ground_truth.get(result.question_id)
            if not expected_sources:
                continue
            hits = 0
            for expected_source in expected_sources:
                found = False
                for ret_src in result.ret_srcs[:k]:
                    if expected_source.file_path == ret_src.file_path:
                        src = expected_source
                        intersection = max(
                            0, min(
                                src.last_character_index,
                                ret_src.last_character_index) - max(
                                    src.first_character_index,
                                    ret_src.first_character_index))
                        union = (
                            src.last_character_index -
                            src.first_character_index) + (
                                ret_src.last_character_index -
                                ret_src.first_character_index) - intersection
                        overlap = intersection / union

                        if overlap >= 0.05:
                            found = True
                            break
                if found:
                    hits += 1
            recall = hits / len(expected_sources)
            recalls.append(recall)
        if not recalls:
            print("No questions available for evaluation")
            return
        return sum(recalls) / len(recalls)
