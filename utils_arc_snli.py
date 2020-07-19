import csv
import glob
import json
import logging
import os
from typing import List
import random
import sys

import tqdm
import pandas as pd
import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

class InputExample(object):
	# A single training/test example for the multiple choice
	# To do changes here
	def __init__(self, example_id, question, contexts, hypothesis, endings, label=None):
		# Conctructs an InputExample.

		'''
		Args:
			example_id : unique_id for the example.
			contexts: List of str. The untokenized texts of the first sequence ( context of corresponding question).
			question: string. The untokenized text of the second sequence (question)
			endings: list of str, multiple choice's options. Its length must be equal to contexts' length.
			label: (Optional) string. The label of the example. This should be specifed for train and dev examples, but not for test examples.
			premise: list of str
			hypothesis : list of str
		'''
		self.example_id = example_id
		self.question = question
		self.contexts = contexts
		self.hypothesis = hypothesis
		self.endings = endings
		self.label = label

# To do changes here
class InputFeatures(object):
	def __init__(self, example_id, choices_features, label):
		self.example_id = example_id
		self.choices_features = [
			{"input_ids" : input_ids, "attention_mask" : attention_mask, "token_type_ids": token_type_ids}
			for input_ids, attention_mask, token_type_ids in choices_features
		]
		self.label = label

class DataProcessor(object):
	# Base class for data converters for multiple choice data sets.

	def get_train_examples(self, data_dir):
		# Gets a collection of InputExamples's for the train set.
		raise NotImplementedError()
	def get_dev_examples(self, data_dir):
		# Gets a collection of InputExampes's for the dev set
		raise NotImplementedError()
	def get_test_examples(self, data_dir):
		# Gets a collection of InputExamples's for the test set
		raise NotImplementedError()

	def get_labels(self):
		# Gets the list of labels for this dataset
		raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f,delimiter='\t'))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples

class SNLIExample(object):
	"""A single training/test example for the ARC dataset."""

	def __init__(self,
				snli_id,
				context_sentences,
				hypothesis,
				label=None):
		self.snli_id = snli_id
		self.context_sentences = context_sentences
		self.hypothesis = hypothesis
		self.label = label

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		l = [
			"snli_id: {}".format(self.snli_id),
			"context_sentences: {}".format(self.context_sentences),
			"hypothesis: {}".format(self.hypothesis),
		]

		if self.label is not None:
		    l.append("label: {}".format(self.label))

		return ", ".join(l)


class SNLIProcessor(DataProcessor):
	# Processor for the ARC data set
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir,"./snli/snli_1.0_train.txt")),type="train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir,"./snli/snli_1.0_dev.txt")), "dev")

	def get_test_examples(self, data_dir):
		logger.info("LOOKING AT {} test".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir,"./snli/snli_1.0_dev.txt")), "test")

	def get_labels(self):
		"""See base class."""
		return ["0", "1", "2"]

	def _read_csv(self, input_file):
		with open(input_file,"r", encoding="utf-8") as f:
			return list(csv.reader(f,delimiter='\t'))

	def _create_examples(self, lines: List[List[str]], type:str):
		# Create examples for training and dev set

		# if type == "train" and lines[0][0] != 'answerKey':
		# 	raise ValueError(
		# 		"For training, the input file must contain a label column."
		# 	)
		# There are two types of labels. They should be normalized
		

		classes = ['contradiction','neutral','entailment']
		labels = lines[9]
		#labels = labels.apply(classes.index)

		examples = []
		i = 0
		for line in lines[1:]:
			hypothesis = [line[6]]
			context_sentences = [line[5]]
			label = line[9]
			examples.append(
			    SNLIExample(snli_id=line[7], context_sentences=context_sentences, hypothesis = hypothesis, label=label))

		return examples

def convert_examples_to_features(
	examples: List[SNLIExample],
	label_list: List[str],
	max_length: int,
	tokenizer: PreTrainedTokenizer,
	pad_token_segment_id=0,
	pad_on_left=False,
	pad_token=0,
	mask_padding_with_zero=True,
	) -> List[InputFeatures]:

	features = []
	max_length = 512

	for example_index, example in enumerate(examples):
		# todo change here
		# context_tokens = tokenizer.tokenize(example.context_sentences)
		context_tokens = []
		hyp_tokens = []
		premise = []
		choices_features = []
		# for context in example.context_sentences:
		# 	context_token = tokenizer.tokenize(context)
		# 	print("context_token{}\n".format(context_token))
		# 	context_tokens.append(context_token)
		# for word in example.hypothesis:
		# 	hyp_token = tokenizer.tokenize(word)
		# 	hyp_tokens.append(hyp_token)

		# We create a copy of the context tokens in order to be
		# able to shrink it according to ending_tokens
		# print("context_tokens is {}\n".format(context_tokens))
		# print("hyp_tokens is {}\n".format(hyp_tokens))
		#context_tokens_choice = context_tokens[:]
		# Modifies `context_tokens_choice` and `ending_tokens` in
		# place so that the total length is less than the
		# specified length.  Account for [CLS], [SEP], [SEP] with
		# "- 3"
		#_truncate_seq(context_tokens_choice, 340)
		_truncate_seq(example.context_sentences, 250)            
		_truncate_seq(example.hypothesis,250)
# 
		cls_segment_id = [2]

		
		pad_token = 0
		# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
		if pad_on_left:
			inputs = tokenizer.encode_plus(example.context_sentences, example.hypothesis, add_special_tokens=True, max_length=max_length,)
			input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
			#input_ids = context_tokens + ["[SEP]"] + hyp_tokens + ["[SEP]"] + ["[CLS]"]
			#input_ids = tokenizer.convert_tokens_to_ids(input_ids)
			#token_type_ids = (len(context_tokens) + 1) * [0] + (len(hyp_tokens) + 2) * [1] + cls_segment_id
		else:
			# input_ids = ["[CLS]"] + context_tokens + ["[SEP]"] + hyp_tokens + ["[SEP]"]
			# input_ids = tokenizer.convert_tokens_to_ids(input_ids)
			# token_type_ids = (len(context_tokens) + 2 ) * [0] + (len(hyp_tokens) + 1) * [1]
			inputs = tokenizer.encode_plus(example.context_sentences, example.hypothesis, add_special_tokens=True, max_length=max_length,)
			input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

		padding_length = max_length - len(input_ids)
		# print("padding_length is {}\n".format(padding_length))
		attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

		if pad_on_left:
			input_ids = ([pad_token] * padding_length) + input_ids
			attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
			token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
		else:
			input_ids = input_ids + ([pad_token] * padding_length)
			# print(len(input_ids))
			attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
			token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length )
			# print(len(input_ids))
			# print("input_ids length is {}\n".format(len(input_ids)))
			assert len(input_ids) == max_length
			assert len(attention_mask) == max_length
			assert len(token_type_ids) == max_length

			choices_features.append((input_ids, attention_mask, token_type_ids))

			label = example.label

		features.append(
			InputFeatures(
			example_id=example.snli_id,
			choices_features=choices_features,
			label=label
			)
		)

	return features

def _truncate_seq(tokens_a, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		if len(tokens_a) > max_length:
			tokens_a.pop()
		else:
			break

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	ctx_max_len = 450
	ans_que_max_len = 42

	total_length = len(tokens_a) + len(tokens_b)
	if total_length <= max_length:
		return

	while True:
		if len(tokens_a) > ctx_max_len:
			tokens_a.pop()
		else:
			break

processors = {"snli" : SNLIProcessor, "race": RaceProcessor, "swag": SwagProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"snli", 3,"race", 4, "swag", 4}

