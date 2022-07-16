## Question Answering with BERTserini

This is the workflow of the BERTserini QA system:

![workflow](fig/BERTserini_workflow.png)

In this project we reproduce
an end-to-end open-domain Question Answering system: BERTserini. It integrates a
BERT reader with the open-source Anserini information retrieval
toolkit built on top of the popular open-source Lucene search
engine. The BERT model is fine-tuned on [SQuAD](https://arxiv.org/abs/1606.05250), a reading
comprehension dataset, consisting of questions+answers pairs
posed on a set of Wikipedia articles.

## Installation

Experiments have been made on Google Colab, using Python 3.7.13 and Torch 1.11.0+cu113. 

The other requirements can be found in the file _requirements.txt_ and you can easily install them via pip
```console
pip install -r requirements.txt
```

## Finetuning

In order to finetune BERT on SQuAD, you can run [finetune.py](bertserini/model/finetune.py).
Use this command to have more informations about the arguments to pass:
```console
python bertserini/model/finetune.py --help
```

## Inference

To answer a question, you can run [inference.py](inference.py).
As for finetuning, type this to have info about command line arguments:
```console
python inference.py --help
```

## Evaluation

If you want to test the pipeline performances, use [evaluation.py](evaluation.py).
```console
python evaluation.py --help
```

## Examples

For example, if you want to answer a question you can run this:
```console
python inference.py --question_text=<INSERT ANSWER> --k=<INSERT NUMBER OF CONTEXTS> --model_path=<INSERT MODEL NAME OR PATH>
```
To use RIDER re-ranking you can just add the argument _--rider_reranking_ when you run _inference.py_ or _evaluation.py_. 