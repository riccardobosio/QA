from bertserini.utils.retriever import Retriever
from bertserini.model.reader import BERT
from bertserini.utils.base import Question
from bertserini.utils.squad import get_best_answer


class BERTserini:
    def __init__(self, model_path, index="enwiki-paragraphs"):
        self.retriever = Retriever(index)
        self.reader = BERT(model_path)

    def retrieve(self, example, k):
        self.question = Question(example['question'])
        self.question.id = example['id']
        self.contexts = self.retriever.retrieve(self.question, k)

    def get_contexts(self):
        return self.contexts

    def answer(self, weight):
        possible_answers = self.reader.predict(self.question, self.contexts)
        # added the following only for rider experiments
        self.all_answers = possible_answers
        answer = get_best_answer(possible_answers, weight)
        pred_answer = {'id': self.question.id, 'prediction_text': answer.text}
        return pred_answer

    def get_question(self):
        return self.question