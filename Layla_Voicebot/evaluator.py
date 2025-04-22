from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_recall, context_precision
from ragas import evaluate
from datasets import Dataset

class Evaluator:
    def evaluate_responses(self, questions, answer, references):
        dataset = Dataset.from_dict({
            "question": [questions],
            "answer": [answer],
            "reference": [references],
            "contexts":[[references]],
        })
        result = evaluate(dataset, metrics=[
            answer_correctness,
            answer_relevancy,
            context_recall,
            context_precision,
            faithfulness
        ], show_progress=True)
        results_df = result.to_pandas()
        return results_df