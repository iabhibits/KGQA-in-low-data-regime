import configparser
import json
import spacy
from spacy.matcher import Matcher
import sys
import timeit
from tqdm import tqdm
import numpy as np
blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


concept_vocab = set()
config = configparser.ConfigParser()
config.read("paths.cfg")
with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_"," "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    return lcs

def load_matcher(nlp):
    config = configparser.ConfigParser()
    config.read("paths.cfg")
    with open(config["paths"]["matcher_patterns"], "r", encoding="utf8") as f:
        all_patterns = json.load(f)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, None, pattern)
    return matcher

def ground_mentioned_concepts(nlp, matcher, s, ans = ""):
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    for match_id, start, end in matches:

        span = doc[start:end].text  # the matched span
        if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
            continue
        original_concept = nlp.vocab.strings[match_id]
        # print("Matched '" + span + "' to the rule '" + string_id)

        if len(original_concept.split("_")) == 1:
            original_concept = list(lemmatize(nlp, original_concept))[0]

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].add(original_concept)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3] #
        for c in shortest:
            if c in blacklist:
                continue
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect)>0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)


    # stop = timeit.default_timer()
    # print('\t Done! Time: ', "{0:.2f} sec".format(float(stop - start_time)))

    # if __name__ == "__main__":
    #     print("Sentence: " + s)
    #     print(mentioned_concepts)
    #     print()
    return mentioned_concepts

def hard_ground(nlp, sent):
    global cpnet_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = "_".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    return res

def match_mentioned_concepts(nlp, sents, answers, batch_id = -1):
    matcher = load_matcher(nlp)

    res = []
    # print("Begin matching concepts.")
    for sid, s in enumerate(sents):
        #print("s is {}\n".format(s))
        #print("sid is {}\n".format(sid))
        a = answers[sid]
        #print("answers is {}\n".format(answers))
        #print("a is {}\n".format(a))
        all_concepts = ground_mentioned_concepts(nlp, matcher, s, a)
        answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
        question_concepts = all_concepts - answer_concepts
        if len(question_concepts)==0:
            # print(s)
            question_concepts = hard_ground(nlp, s) # not very possible
        if len(answer_concepts)==0:
            #print(answer)
            answer_concepts = hard_ground(nlp, answers) # some case
            print(answer_concepts)

        res.append({"sent": s, "ans": a, "qc": list(question_concepts), "ac": list(answer_concepts)})
    return res
def match_mentioned_concepts_context(nlp,context_sents,hypo_sents,answer):
    matcher = load_matcher(nlp)

    res = []
    all_context_concepts = ground_mentioned_concepts(nlp, matcher, context_sents,answer)
    all_hypo_concepts = ground_mentioned_concepts(nlp,matcher,hypo_sents,answer)
    answer_concepts = ground_mentioned_concepts(nlp, matcher, answer)
    hypo_concepts = all_hypo_concepts - answer_concepts
    context_concepts = all_context_concepts - answer_concepts#
    #print("context concepts is {}\n".format(context_concepts))
    #print("hyp concepts is {}\n".format(hypo_concepts))
    if len(context_concepts)==0:
        #print(s)
        context_concepts = hard_ground(nlp, context_sents) # not very possible
    if len(hypo_concepts)==0:
        #print(s)
        hypo_concepts = hard_ground(nlp, hypo_sents)
    if len(answer_concepts)==0:
        #print(a)
        answer_concepts = hard_ground(nlp, answer) # some case
        #print(answer_concepts)
    res.append({"sent":" ", "option": answer, "premise": list(context_concepts),
                "hypothesis": list(hypo_concepts),"ans":list(answer_concepts)})
    return res

def process(filename, batch_id=-1):


    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    sents = []
    answers = []
    with open(filename, 'r') as f:
        lines = f.read().split("\n")


    for line in tqdm(lines, desc="loading file"):
        if line == "":
            continue
        j = json.loads(line)
        for statement in j["statements"]:
            sents.append(statement["statement"])
        for answer in j["question"]["choices"]:
            answers.append(answer["text"])


    if batch_id >= 0:
        output_path = filename + ".%d.mcp" % batch_id
        batch_sents = list(np.array_split(sents, 100)[batch_id])
        batch_answers = list(np.array_split(answers, 100)[batch_id])
    else:
        output_path = filename + ".mcp"
        batch_sents = sents
        batch_answers = answers

    res = match_mentioned_concepts(nlp, sents=batch_sents, answers=batch_answers, batch_id=batch_id)
    with open(output_path, 'w') as fo:
        json.dump(res, fo)



def test(sent):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    #res = match_mentioned_concepts(nlp, sents=["Watch television do children require to grow up healthy."], answers=["watch television"])
    res = match_mentioned_concepts(nlp, sents=sent, answers=["watch television"])
    print(res)

# "sent": "Watch television do children require to grow up healthy.", "ans": "watch television",
if __name__ == "__main__":
    process(sys.argv[1], int(sys.argv[2]))

# test()
