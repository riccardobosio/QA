from typing import List, Union, Optional, Mapping, Any
import abc

__all__ = ['Question', 'Context', 'Reader', 'Answer', 'TextType']


TextType = Union['Question', 'Context', 'Answer']

class Question:
    def __init__(self, text, id=None, language="en"):
        self.text = text
        self.id = id
        self.language = language

# forse inutili
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<Question:{}>".format(self.text)

class Context:
    def __init__(self, text, title="", language="en", metadata=None, score=0):
        self.text = text
        self.title = title
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<Passage:{},\n score:{}>".format(self.text, self.score)

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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<Answer: {}, score:{}, ctx_score:{}, total_score:{}>".format(self.text, self.score, self.ctx_score, self.total_score)

    def aggregate_score(self, weight):
        self.total_score = weight*self.score + (1-weight)*self.ctx_score