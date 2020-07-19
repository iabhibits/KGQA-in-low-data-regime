from __future__ import absolute_import
from grounding_concepts import *
import re
import string
import pandas as pd
import csv
from io import open

import argparse
import numpy as np
from tqdm import tqdm, trange
import os, sys, shutil
import time
import gc
from contextlib import contextmanager
from pathlib import Path
import random

class ArcExample(object):
    """A single training/test example for the ARC dataset."""

    def __init__(self,
                 arc_id,
                 context_sentences,
                 hypothesis,
                 question,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.arc_id = arc_id
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

def read_arc_examples(input_file, is_training=True):
    '''
    :returns :
    training dataset.
    list of a single training/testing example for the ARC dataset.
    A single ARC example is a object with following format :
    (context sentence, hypothesis, option a, option b , ... , label)

    '''
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)

    if is_training and lines[0][0] != 'answerKey':
        raise ValueError(
            "For training, the input file must contain a label column."
        )

    examples = []
    for line in lines[1:]:
        context_sentences = [line[5], line[6], line[7], line[8]]
        hypothesis = [line[9], line[10], line[11], line[12]]
        label = ord(line[0]) - ord('A') if is_training else None

        '''examples.append(
                                    ArcExample(arc_id=line[-2], context_sentences=context_sentences, hypothesis = hypothesis, question=line[-1], ending_0=line[1],
                                               ending_1=line[2], ending_2=line[3], ending_3=line[4], label=label))
                        '''
        examples.append(
                                    ArcExample(arc_id=line[-2], context_sentences=context_sentences, hypothesis = hypothesis, question=line[-3], ending_0=line[1],
                                               ending_1=line[2], ending_2=line[3], ending_3=line[4], label=label))
    return examples


def proces_arc(args,data):
    nlp = spacy.load('en_core_web_sm',disable=['ner','parser','textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    output_path = args.output_dir+"arc_concepts"+str(args.start)+"_"+str(args.end)+".mcp"
    concepts = []
    examples = data[args.start:args.end]
    for i,example in tqdm(enumerate(examples),total=len(examples), desc="grounding batch_id:%d"):
        x = []
        for j in tqdm(range(4)):
            res = match_mentioned_concepts_context(nlp,context_sents=example.context_sentences[j][:1024],
                                                   hypo_sents=example.hypothesis[j],
                                                   answer=example.endings[j])
            x.append(res)
        concepts.append(x)
    with open(output_path, 'w') as fo:
        json.dump(concepts, fo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../datasets/Train_Final_complete.csv",
        type=str,
        help="The input data dir. Should contain the .csv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default="../datasets/arc_data/",
        type=str,
        help="The output data dir.",
    )
    parser.add_argument( "--start", default=0, type=int, help="starting index of arc question")
    parser.add_argument("--end",default=250,type=int, help="last index of arc question")
    args = parser.parse_args()


    #data = pd.read_csv(args.data_dir)
    examples = read_arc_examples(args.data_dir)

    proces_arc(args,examples)


if __name__ == '__main__':
    main()
