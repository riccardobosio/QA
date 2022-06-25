from models.reader import BERTModule
from utils.pyserini import Retriever
from utils.base import Question, Context
import argparse
from models.reader import BERTModule
from utils.squad import get_best_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Define inference parameters.')
    parser.add_argument('--text', type=str, help='Pass the text of the question.')
    parser.add_argument('--model_path', type=str, help='Finetuned reader path.')

    args = parser.parse_args()

    retriever = Retriever('wikipedia-dpr')
    reader = BERTModule(model_path=args.model_path)
    question = Question(args.text)
    contexts = retriever.retrieve(question=question)
    all_answers = reader.predict(question, contexts)
    answer = get_best_answer(all_answers) # si può modificare il weight (non passandolo si usa il default 0.5)
    pred_answer = {'prediction_text': answer.text, 'id': question.id}
    print(pred_answer)


