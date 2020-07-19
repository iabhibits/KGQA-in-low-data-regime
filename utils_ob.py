import csv
import torch
import glob
import json
import logging
import os
from typing import List
import random
import json
import timeit
import pickle
#import dgl
import networkx as nx
import random
import numpy as np

import tqdm

from transformers.tokenization_utils import PreTrainedTokenizer
from kg_loader import KG
from graph_utils import GraphEncoder

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
			{"input_ids" : input_ids,
			 "attention_mask" : attention_mask, 
			 "token_type_ids": token_type_ids
			 }
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
            return list(csv.reader(f))

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

class OBExample(object):
	"""A single training/test example for the ARC dataset."""

	def __init__(self,
				ob_id,
				context_sentences,
				hypothesis,
				question,
				ending_0,
				ending_1,
				ending_2,
				ending_3,
				label=None):
		self.ob_id = ob_id
		self.context_sentences = context_sentences
		self.hypothesis = hypothesis
		self.question = question
		self.endings = [
			ending_0,
			ending_1,
			ending_2,
			ending_3,
		]
		self.label = label

	def __str__(self):
		return self.__repr__()

	def __repr__(self):
		l = [
			"ob_id: {}".format(self.ob_id),
			"context_sentences: {}".format(self.context_sentences),
			"hypothesis: {}".format(self.hypothesis),
			"question: {}".format(self.question),
			"ending_0: {}".format(self.endings[0]),
			"ending_1: {}".format(self.endings[1]),
			"ending_2: {}".format(self.endings[2]),
			"ending_3: {}".format(self.endings[3]),
		]

		if self.label is not None:
		    l.append("label: {}".format(self.label))

		return ", ".join(l)


class OBProcessor(DataProcessor):
	# Processor for the ARC data set
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {} train".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir,"train_openbook_new.csv")), "train")

	def get_dev_examples(self, data_dir):
		logger.info("LOOKING AT {} dev".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir,"dev_openbook_new.csv")), "dev")

	def get_test_examples(self, data_dir):
		logger.info("LOOKING AT {} test".format(data_dir))
		return self._create_examples(self._read_csv(os.path.join(data_dir,"test_openbook_new.csv")), "test")

	def get_labels(self):
		"""See base class."""
		return ["0", "1", "2", "3"]

	def _read_csv(self, input_file):
		with open(input_file,"r", encoding="utf-8") as f:
			return list(csv.reader(f))
		# df = pd.read_csv(
		# 	input_file,
		# 	header=0,
		# 	skiprows=lambda i: i>0 and random.random() > p
		# )
		# return list(df)

	def _create_examples(self, lines: List[List[str]], type:str):
		# Create examples for training and dev set

		if type == "train" and lines[0][1] != 'answerkey':
			raise ValueError(
				"For training, the input file must contain a label column."
			)
		# There are two types of labels. They should be normalized
		def normalize(truth):
			if truth in "ABCD":
			    return ord(truth) - ord("A")
			elif truth in "1234":
			    return int(truth) - 1
			else:
			    logger.info("truth ERROR! %s", str(truth))
			    return None

		examples = []
		for line in lines[1:]:
			context_sentences = [line[6], line[7], line[8], line[9]]
			hypothesis = [line[2], line[3], line[4], line[5]]
			label = ord(line[1]) - ord('A')

			examples.append(
			    OBExample(ob_id=line[0], context_sentences=context_sentences, hypothesis = hypothesis, question=line[10], ending_0=line[11],
			               ending_1=line[12], ending_2=line[13], ending_3=line[14], label=label))

		return examples

def read_graph_example(args,typ):
	nxgs = []
	dgs = []
	start_time = timeit.default_timer()
	if typ == 'train':
		graph_ngx_file = args.path_to_graph + "ob_sim_train_graph.pnxg"
	elif typ == 'val':
		graph_ngx_file = args.path_to_graph + "ob_sim_dev_graph.pnxg"
	elif typ == 'test':
		graph_ngx_file = args.path_to_graph + "ob_sim_test_graph.pnxg"
	print("loading paths from %s" % graph_ngx_file)
	with open(graph_ngx_file, 'r') as fr:
		for line in fr.readlines():
			line = line.strip()
			nxgs.append(line)
	print('\t Done! Time: ', "{0:.2f} sec".format(float(timeit.default_timer() - start_time)))
	if typ == 'train':
		save_file = args.path_to_graph+"ob_sim_train_graph.pnxg.dgl.pk"
	elif typ == 'val':
		save_file = args.path_to_graph +"ob_sim_dev_graph.pnxg.dgl.pk"
	elif typ == 'test':
		save_file = args.path_to_graph +"ob_sim_test_graph.pnxg.dgl.pk"
	reload=True

	if reload and os.path.exists(save_file):
		import gc
		print("loading pickle for the dgl", save_file)
		start_time = timeit.default_timer()
		with open(save_file, 'rb') as handle:
			gc.disable()
			dgs = pickle.load(handle)
			gc.enable()
		print("finished loading in %.3f secs" % (float(timeit.default_timer() - start_time)))
	else:
		for index, nxg_str in tqdm(enumerate(nxgs), total=len(nxgs)):
			nxg = nx.node_link_graph(json.loads(nxg_str))
			dg = dgl.DGLGraph(multigraph=True)
			# dg.from_networkx(nxg, edge_attrs=["rel"])
			dg.from_networkx(nxg)
			cids = [nxg.nodes[n_id]['cid']+1 for n_id in range(len(dg))] # -1 --> 0 and 0 stands for a palceholder concept
			# rel_types = [nxg.edges[u, v, r]["rel"] + 1 for u, v, r in nxg.edges]  # 0 is used for

			# print(line)
			# node_types = [mapping_type[nxg.nodes[n_id]['type']] for n_id in range(len(dg))]
			# edge_weights = [nxg.edges[u, v].get("weight", 0.0) for u, v in nxg.edges]  # -1 is used for the unk edges
			# dg.edata.update({'weights': torch.FloatTensor(edge_weights)})

			# dg.edata.update({'rel_types': torch.LongTensor(rel_types)})

			dg.ndata.update({'cncpt_ids': torch.LongTensor(cids)})
			dgs.append(dg)
		num_choice = 4
		nxgs = list(zip(*(iter(nxgs),) * num_choice))
		dgs = list(zip(*(iter(dgs),) * num_choice))

	return dgs

# def convert_to_tensor(data):
# 	lists = []
# 	for elements in data:
# 		element = torch.stack(elements)
# 		lists.append(element)
# 	lists = torch.stack(lists)
# 	return lists


def convert_examples_to_features(args,
	examples: List[OBExample],
	label_list: List[str],
	max_length: int,
	tokenizer: PreTrainedTokenizer,
	typ,
	pad_token_segment_id=0,
	pad_on_left=False,
	pad_token=0,
	mask_padding_with_zero=True,
	) -> List[InputFeatures]:

	features = []
	max_seq_length = args.max_seq_length
	#dgs = read_graph_example(args,typ)
	if typ == 'train':
		mat_file = args.path_to_graph+"train_matrix.pt"
		con_file = args.path_to_graph+"train_concept.pt"
	elif typ == 'val':
		mat_file = args.path_to_graph+"dev_matrix.pt"
		con_file = args.path_to_graph+"dev_concept.pt"
	elif typ == 'test':
		mat_file = args.path_to_graph+"test_matrix.pt"
		con_file = args.path_to_graph+"test_concept.pt"
	matrix = torch.load(mat_file)
	tconcept = torch.load(con_file)

	for example_index, example in enumerate(examples):
		# todo change here
		# context_tokens = tokenizer.tokenize(example.context_sentences)
		context_tokens = []
		hypothesis = []
		premise = []
		for context in example.context_sentences:
			context_token = tokenizer.tokenize(context)
			context_tokens.append(context_token)
		#hypo_tokens = graph_encoder.encode(example.hypothesis)
		#prem_tokens = graph_encoder.encode(example.context_sentences)
		question_tokens = tokenizer.tokenize(example.question)

		choices_features = []
		for ending_index, ending in enumerate(example.endings):
			# We create a copy of the context tokens in order to be
			# able to shrink it according to ending_tokens
			context_tokens_choice = context_tokens[ending_index][:]
			ending_tokens = tokenizer.tokenize(ending)
			# Modifies `context_tokens_choice` and `ending_tokens` in
			# place so that the total length is less than the
			# specified length.  Account for [CLS], [SEP], [SEP] with
			# "- 3"
			if args.max_seq_length == 512:
				_truncate_seq(context_tokens_choice, 340)
			elif args.max_seq_length == 384:
				_truncate_seq(context_tokens_choice, 210)            
			else:
				seqence_len = args.max_seq_length - (127+42+3)
				_truncate_seq(context_tokens_choice,seqence_len)
				
			_truncate_seq(question_tokens,127)
			_truncate_seq(ending_tokens, 42)

			#l = len(prem_tokens[ending_index])
			#m = len(hypo_tokens[ending_index])

			#max_ent_pre = 262
			#max_ent_hyp = 83

			#padpre = [kg.num_entities] * (max_ent_pre - min(l,max_ent_pre))
			#padhyp = [kg.num_entities] * (max_ent_hyp - min(m,max_ent_hyp))

			context_tokens_choice = context_tokens_choice

			cls_segment_id = [2]

			# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
			if pad_on_left:
				input_ids = context_tokens_choice + question_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"] + ["[CLS]"]
				input_ids = tokenizer.convert_tokens_to_ids(input_ids)
				token_type_ids = (len(context_tokens_choice) + len(question_tokens) + 1) * [0] + (len(ending_tokens) + 2) * [1] + cls_segment_id
			else:
				input_ids = ["[CLS]"] + context_tokens_choice + question_tokens + ["[SEP]"] + ending_tokens + ["[SEP]"]
				input_ids = tokenizer.convert_tokens_to_ids(input_ids)
				token_type_ids = (len(context_tokens_choice) + len(question_tokens) + 2 ) * [0] + (len(ending_tokens) + 1) * [1]

			padding_length = max_length - len(input_ids)
			attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

			if pad_on_left:
				input_ids = ([pad_token] * padding_length) + input_ids
				attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
				token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
				# gpre = padpre + prem_tokens[ending_index][:min(l,max_ent_pre)]
				# ghyp = padhyp + hypo_tokens[ending_index][:min(m,max_ent_hyp)]
				#gpre = prem_tokens[ending_index][:min(l,max_ent_pre)] + padpre
				#ghyp = hypo_tokens[ending_index][:min(m,max_ent_hyp)] + padhyp
			else:
				input_ids = input_ids + ([pad_token] * padding_length)
				attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
				token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length )
				#gpre = prem_tokens[ending_index][:min(l,max_ent_pre)] + padpre
				#ghyp = hypo_tokens[ending_index][:min(m,max_ent_hyp)] + padhyp

			#ggraph = dgs[example_index][ending_index]
			#concept = tconcept[example_index][ending_index]
			#adj_matrix = matrix[example_index][ending_index]
			# cnt = 1
			# if (cnt==1):
			# 	print("length of Input ids {} \n".format(len(input_ids)))
			# 	print("length of attention ids {} \n".format(len(attention_mask)))
			# 	print("length of token type ids {} \n".format(len(token_type_ids)))
			# 	cnt += 1
			assert len(input_ids) == max_seq_length
			assert len(attention_mask) == max_seq_length
			assert len(token_type_ids) == max_seq_length

			choices_features.append((input_ids, attention_mask, token_type_ids))
			#input_ids, attention_mask, token_type_ids, gpre, ghyp,  concept, adj_matrix

			label = example.label

		features.append(
			InputFeatures(
			example_id=example.ob_id,
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



processors = {"ob" : OBProcessor, "race": RaceProcessor, "swag": SwagProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"ob", 4,"race", 4, "swag", 4}
