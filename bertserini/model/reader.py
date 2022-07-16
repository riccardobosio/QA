from transformers import AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult
import torch
from torch.utils.data import DataLoader, SequentialSampler
from bertserini.utils.base import Answer
from bertserini.utils.squad import SquadExample, compute_predictions_logits


class BERT:
    def __init__(self, model_name, tokenizer_name=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(self.device).eval()
        if tokenizer_name is None:
            tokenizer_name = model_name
        self.tokenizer_name = tokenizer_name

    def predict(self, question, contexts):
        model = self.model
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)
        device = self.device
        model.to(device)

        examples = []
        for idx, ctx in enumerate(contexts):
            examples.append(
                SquadExample(
                    qas_id=idx,
                    question_text=question.text,
                    context_text=ctx.text,
                    answer_text=None,
                    start_position_character=None,
                    title="",
                    is_impossible=False,
                    answers=[],
                    language=ctx.language
                )
            )

        features, dataset = squad_convert_examples_to_features(
                  examples=examples,
                  tokenizer=tokenizer,
                  max_seq_length=384,
                  doc_stride=128,
                  max_query_length=64,
                  is_training=False,
                  return_dataset="pt",
                  threads=1
                )

        eval_sampler = SequentialSampler(dataset)
        eval_dataset = DataLoader(dataset, sampler=eval_sampler, batch_size=32)

        all_results = []
        for batch in eval_dataset:
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                feature_indices = batch[3]

                outputs = model(**inputs)
                for i, feature_index in enumerate(feature_indices):
                        eval_feature = features[feature_index.item()]
                        unique_id = int(eval_feature.unique_id)
                        start_logits, end_logits = outputs.start_logits[i].detach().cpu().tolist(), outputs.end_logits[i].detach().cpu().tolist()
                        result = SquadResult(unique_id, start_logits, end_logits)
                        all_results.append(result)

        answers, n_best = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            n_best_size=10,
            max_answer_length=64,
            do_lower_case=True,
            output_prediction_file=False,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=1,
            version_2_with_negative=False,
            null_score_diff_threshold=0,
            tokenizer=tokenizer
        )
        
        all_answers = []
        for idx, ans in enumerate(n_best):
            all_answers.append(Answer(
                text=answers[ans][0],
                score=answers[ans][1],
                ctx_score=contexts[idx].score,
                language='en'
            ))

        return all_answers