import argparse
import numpy as np
from datasets import load_metric, load_dataset
from bertserini.BERTserini import BERTserini
from bertserini.utils.rider import rerank


def evaluate(model_path, num_val_examples):
    bertserini = BERTserini(model_path)
    squad = load_dataset("squad")
    metric = load_metric("squad")
    indexes = np.random.uniform(low=0, high=squad["validation"].num_rows, size=num_val_examples)
    val_examples = squad["validation"].select(indexes)
    predicted_answers = []
    ground_truth = []
    for i, question in enumerate(val_examples):
        print(f"\nQuestion number {i}:\n {question['question']}")
        if args.rider_ranking:
          bertserini.retrieve(question, k=20)
          all_contexts = bertserini.contexts
          print('-------')
          print('Contexts before re-ranking:')
          for ctx in all_contexts[:args.k]:
            print(ctx.text)
          bertserini.contexts = all_contexts[:args.k]
          answer = bertserini.answer(args.weight)
          print('-------')
          print('Answer before re-ranking')
          print(answer['prediction_text'])
          top_N_answers = bertserini.get_top_n_answers(args.weight, N=5)
          reranked_contexts = rerank(all_contexts, args.k, top_N_answers)
          print('-------')
          print('Contexts after re-ranking')
          for ctx in reranked_contexts:
            print(ctx.text)
          bertserini.contexts = reranked_contexts
          answer = bertserini.answer(args.weight)
          print('-------')
          print('Answer after re-ranking')
          print(answer['prediction_text'])
        else:
          bertserini.retrieve(question, k=args.k)
          answer = bertserini.answer(args.weight)

        predicted_answers.append(answer)
        ground_truth.append({"id": val_examples["id"][i], "answers": val_examples["answers"][i]})
    
    result = metric.compute(predictions=predicted_answers, references=ground_truth)
    print(f"\nThese are the evaluation results: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="Reader path or name.")
    parser.add_argument('--k', type=int, default=10, help="Number of contexts to retrieve.")
    parser.add_argument('--weight', type=float, default=0.5, help="Weight used by the linear interpolation to compute final score.")
    parser.add_argument('--num_val_examples', type=int, help="Number of Squad validation examples to use.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rider_ranking', action='store_true', help="Whether to use or not contexts re-ranking")
    args = parser.parse_args()

    np.random.seed(args.seed)

    evaluate(args.model_path, args.num_val_examples)