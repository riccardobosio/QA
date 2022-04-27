# Followed this tutorial to write the code https://huggingface.co/course/chapter7/7?fw=tf

from tensorflow import keras
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering, TFAutoModelForQuestionAnswering
from datasets import load_dataset
from transformers import AutoTokenizer
import collections
import numpy as np
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import DefaultDataCollator
from transformers import create_optimizer
from transformers.keras_callbacks import PushToHubCallback
import tensorflow as tf


def compute_metrics(start_logits, end_logits, features, examples, max_answer_length=30, n_best=20):
    metric = load_metric("squad")
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

class BERTModule(tf.Module):

    def __init__(self, model_name):

        #self.model = TFBertForQuestionAnswering.from_pretrained(model_name)
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        #controllare se serve!!
        self.all_results=[]

        self.raw_dataset = load_dataset("squad")

        #forse inutili
        self.raw_train_dataset = self.raw_dataset["train"]
        self.raw_validate_dataset = self.raw_dataset["validation"]

        self.train_dataset = self.raw_dataset["train"].map(
            self.preprocess_training_examples,
            batched=True,
            remove_columns=self.raw_dataset["train"].column_names,
        )
        #add this to see the lenghts
        #print(len(self.raw_dataset["train"]), len(self.train_dataset))

        self.validation_dataset = self.raw_dataset["validation"].map(
            self.preprocess_validation_examples,
            batched=True,
            remove_columns=self.raw_dataset["validation"].column_names,
        )
        # add this to see the lenghts
        #len(self.raw_dataset["validation"]), len(self.validation_dataset)

        self.data_collator = DefaultDataCollator(return_tensors="tf")

        self.tf_train_dataset = self.train_dataset.to_tf_dataset(
            columns=[
                "input_ids",
                "start_positions",
                "end_positions",
                "attention_mask",
                "token_type_ids",
            ],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=16,
        )
        self.tf_eval_dataset = self.validation_dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "token_type_ids"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=16,
        )

    def preprocess_training_examples(self, examples, max_length=384, stride=128):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(self, examples, max_length=384, stride=128):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    def validate(self):

        eval_set_for_model= self.validation_dataset.remove_columns(["example_id", "offset_mapping"])
        eval_set_for_model.set_format("numpy")
        batch = {k: eval_set_for_model[k] for k in eval_set_for_model.column_names}

        outputs = self.model(**batch)
        start_logits = outputs.start_logits.numpy()
        end_logits = outputs.end_logits.numpy()

        compute_metrics(start_logits, end_logits, self.validation_dataset, self.raw_validate_dataset)

    def train(self):
        # The number of training steps is the number of samples in the dataset, divided by the batch size then multiplied
        # by the total number of epochs. Note that the tf_train_dataset here is a batched tf.data.Dataset,
        # not the original Hugging Face Dataset, so its len() is already num_samples // batch_size.
        num_train_epochs = 3
        num_train_steps = len(self.tf_train_dataset) * num_train_epochs
        optimizer, schedule = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=0,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
        )
        self.model.compile(optimizer=optimizer)

        # Train in mixed-precision float16
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # We're going to do validation afterwards, so no validation mid-training
        self.model.fit(self.tf_train_dataset, epochs=num_train_epochs) #, callbacks=[callback])

        predictions = self.model.predict(self.tf_eval_dataset)
        compute_metrics(
            predictions["start_logits"],
            predictions["end_logits"],
            self.validation_dataset,
            self.raw_datasets["validation"],
        )

    def print_data(self):
        print(self.raw_dataset)
        print(self.raw_train_dataset)


if __name__ == "__main__":
    bert_module = BERTModule("bert-base-cased")
    #bert_module.print_data()
    #bert_module.validate()
    bert_module.train()