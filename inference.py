import argparse
import numpy as np
from bertserini.BERTserini import BERTserini
from bertserini.utils.rider import rerank


def infer(args):
    example_question = {"id": np.random.randint(0,10000000), "question": args.question_text}
    bertserini = BERTserini(args.model_path)

    if args.rider_ranking:
        bertserini.retrieve(example_question, k=20)
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
        bertserini.retrieve(example_question, k=args.k)
        answer = bertserini.answer(args.weight)
        print(f"The answer is:\n{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_text", type=str, help="Question to be answered.")
    parser.add_argument("--model_path", type=str, help="Path or name of the Reader.")
    parser.add_argument("--k", type=int, default=10, help="Number of context to retrieve.")
    parser.add_argument("--weight", type=str, default=0.5, help="Weight used by the linear interpolation to compute final score.")
    parser.add_argument("--rider_ranking", action='store_true', help="Whether to use or not contexts re-ranking")
    args = parser.parse_args()

    infer(args)