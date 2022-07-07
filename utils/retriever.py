from pyserini.search.lucene import LuceneSearcher
from bertserini.utils.base import Context


class Retriever:
    def __init__(self, index):
        self.searcher = LuceneSearcher.from_prebuilt_index(index)

    def retrieve(self, question, results=10):
        try:
            hits = self.searcher.search(question.text, k=results)
        except ValueError as e:
            print(f"Failure while retrieving: {question.text}, {e}")

        contexts = []
        for i in range(0, len(hits)):
            id = hits[i].docid
            print(f"Context id: {id}\n")
            score = hits[i].score
            document = self.searcher.doc(hits[i].docid)
            text = document.raw()
            print(document.raw())
            language = 'en'
            parag = Context(id, score, text, language)
            contexts.append(parag)
        return contexts