import json

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .Models import AnsweredQuestion, RagDataset, StudentSearchResults


class EvaluatorError(Exception):
    """A custom error for evaluation"""
    pass


class EvaluationResult(BaseModel):
    """A pydantic model to validate evaluation results"""
    recall1: float
    recall3: float
    recall5: float
    recall10: float
    questions_evaluated: int


class Evaluator:
    def evaluate(self, student_search_results_path: str, dataset_path: str
                 ) -> EvaluationResult:
        """Load the student's search results, calculates recall
           for different values of k and returns the result"""
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_results = (
                    StudentSearchResults.model_validate(json.load(f)))
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValueError):
            raise EvaluatorError("[ERROR]: Could not load evaluation data")

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
            raise EvaluatorError('Could not evaluate the resuls.')

        return results

    @staticmethod
    def get_recall(
        student_results: StudentSearchResults,
        dataset: RagDataset,
            k: int) -> float:
        """
        Calculates an average recall@k over the whole student's dataset
        compared to the ground-truth dataset. Returns the score.
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
            return 0
        return sum(recalls) / len(recalls)
