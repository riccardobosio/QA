class Question:
    def __init__(self, text, id=None, language="en"):
        self.text = text
        self.id = id
        self.language = language

class Context:
    def __init__(self, id, score, text, language="en"):
        self.id = id
        self.text = text
        self.language = language
        self.score = score

class Answer:
    def __init__(self, text, language="en", metadata=None, score=0, ctx_score=0, total_score=0):
        self.text = text
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.ctx_score = ctx_score
        self.total_score = total_score

    def aggregate_score(self, weight):
        self.total_score = weight*self.score + (1-weight)*self.ctx_score