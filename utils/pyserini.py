from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from typing import List
from transformers.data.processors.squad import SquadExample
from base import Context, Question


class Retriever:
    def __init__(self, index, k1=0.9, b=0.4, language="en"):
        self.searcher = SimpleSearcher.from_prebuilt_index(index)
        self.searcher.set_bm25(k1, b)
        self.searcher.object.setLanguage(language)

    def retrieve(self, question: Question, results=20):
        try:
            hits = self.searcher.search(question.text, k=results)
        except ValueError as e:
            print(f"Failure while retrieving: {question.text}, {e}")

        return self.hits_to_contexts(hits)

    def hits_to_contexts(self, hits, language="en", field="raw", blacklist=[]):
        contexts = []
        for hit in hits:
            text = hit.raw
            for s in blacklist:
                if s in text:
                    continue
# valutare se aggiungere metadat
            metadata = {}
            contexts.append(Context(text=text, metadata=metadata, score=hit.score))

        return contexts

    def convert_to_squad(self, question, contexts):
        examples = []
        for idx, ctx in enumerate(contexts):
            examples.append(SquadExample(
                qas_id=idx,
                question_text=question.text,
                context_text=ctx.text,
                answer_text=None,
                start_position_character=None,
                title="",
                is_impossible=False,
                answers=[],
            ))
        return examples