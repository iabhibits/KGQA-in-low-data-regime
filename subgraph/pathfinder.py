from __future__ import absolute_import
import configparser
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import numpy as np
import argparse


config = configparser.ConfigParser()
config.read("paths.cfg")


cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2relation = None
id2concept = None




def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")

def load_cpnet():
    global cpnet,concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def get_edge(src_concept, tgt_concept):
    global cpnet, concept2id, relation2id, id2relation, id2concept
    rel_list = cpnet[src_concept][tgt_concept]
    # tmp = [rel_list[item]["weight"] for item in rel_list]
    # s = tmp.index(min(tmp))
    # rel = rel_list[s]["rel"]
    return list(set([rel_list[item]["rel"] for item in rel_list]))

# source and target is text
def find_paths(source, target, ifprint = False):
    try:
        global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
        s = concept2id[source]
        t = concept2id[target]

        # try:
        #     lenth, path = nx.bidirectional_dijkstra(cpnet, source=s, target=t, weight="weight")
        #     # print(lenth)
        #     print(path)
        # except nx.NetworkXNoPath:
        #     print("no path")
        # paths = [path]

        if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
            return
        # paths =
        all_path = []
        all_path_set = set()

        for max_len in range(1, 3):
            for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=max_len):
                path_str = "-".join([str(c) for c in p])
                if path_str not in all_path_set:
                    all_path_set.add(path_str)
                    all_path.append(p)
                if len(all_path) >= 3:  # top shortest 300 paths
                    break
            if len(all_path) >= 3:  # top shortest 300 paths
                break

        # all_path = [[int(c) for c in p.split("-")] for p in list(set(["-".join([str(c) for c in p]) for p in all_path]))]
        # print(len(all_path))
        all_path.sort(key=len, reverse=False)
        pf_res = []
        for p in all_path:
            # print([id2concept[i] for i in p])
            rl = []
            for src in range(len(p) - 1):
                src_concept = p[src]
                tgt_concept = p[src + 1]

                rel_list = get_edge(src_concept, tgt_concept)
                rl.append(rel_list)
                if ifprint:
                    rel_list_str = []
                    for rel in rel_list:
                        if rel < len(id2relation):
                            rel_list_str.append(id2relation[rel])
                        else:
                            rel_list_str.append(id2relation[rel - len(id2relation)]+"*")
                    print(id2concept[src_concept], "----[%s]---> " %("/".join(rel_list_str)), end="")
                    if src + 1 == len(p) - 1:
                        print(id2concept[tgt_concept], end="")
            if ifprint:
                print()

            pf_res.append({"path": p, "rel": rl})
        return pf_res
    except:
        path_ = []
        return path_


def process(filename, batch_id=-1):
    pf = []
    output_path = filename + ".%d" % (batch_id) + ".pf"
    import os
    if os.path.exists(output_path):
        print(output_path + " exists. Skip!")
        return

    load_resources()
    load_cpnet()
    with open(filename, 'r') as fp:
        mcp_data = json.load(fp)
        mcp_data = list(np.array_split(mcp_data, 100)[batch_id])

        for item in tqdm(mcp_data, desc="batch_id: %d "%batch_id):
            acs = item["ac"]
            qcs = item["qc"]
            pfr_qa = []  # path finding results
            for ac in acs:
                for qc in qcs:
                    pf_res = find_paths(qc, ac)
                    pfr_qa.append({"ac":ac, "qc":qc, "pf_res":pf_res})
            pf.append(pfr_qa)

    with open(output_path, 'w') as fi:
        json.dump(pf, fi)
 

# process(sys.argv[1], int(sys.argv[2]))
#

#load_resources()
#load_cpnet()
# find_paths("fill", "fountain_pen", ifprint=True)
# print("--------")
# find_paths("write", "fountain_pen", ifprint=True)
# print("--------")
# find_paths("write", "pen", ifprint=True)
#find_paths("bottle", "liquor", ifprint=True)

#print();print();print();print();print();


#find_paths("cashier", "store", ifprint=True)
def process_arc(args,data):
    ex = data[args.start:args.end]
    pf = []
    for e in tqdm(ex):
        for item in e:
            gpre = item["premise"]
            ghyp = item["hypothesis"]
            ans = item["ans"]
            pfr_gpre = []  # path finding results
            pfr_hyp = []
            for pre in gpre:
                for ac in ans:
                    gpf_res = find_paths(pre, ac)
                    #print(pf_res)
                    pfr_gpre.append({"pre":pre, "ac":ac, "path":gpf_res})
            for hyp in ghyp:
                for answer in ans:
                    hpf_res = find_paths(hyp,answer)
                    pfr_hyp.append({"hyp":hyp,"hac":answer, "path":hpf_res})
            combine =[pfr_gpre,pfr_hyp]
            pf.append(combine)
    with open(args.output_dir+"arc_50_path_"+str(args.typ)+str(args.start)+"_"+str(args.end)+".mcp","w") as fp:
        json.dump(pf,fp)
def process_ob(args,data):
    ex = data[args.start:args.end]
    pf = []
    for e in tqdm(ex):
        for item in e:
            gpre = item[0]["premise"]
            ghyp = item[0]["hypothesis"]
            ans = item[0]["ans"]
            pfr_gpre = []  # path finding results
            pfr_hyp = []
            for pre in gpre:
                for ac in ans:
                    gpf_res = find_paths(pre, ac)
                    #print(pf_res)
                    pfr_gpre.append({"pre":pre, "ac":ac, "path":gpf_res})
            for hyp in ghyp:
                for answer in ans:
                    hpf_res = find_paths(hyp,answer)
                    pfr_hyp.append({"hyp":hyp,"hac":answer, "path":hpf_res})
            combine =[pfr_gpre,pfr_hyp]
            pf.append(combine)
    with open(args.output_dir+"ob_path_"+str(args.typ)+str(args.start)+"_"+str(args.end)+".mcp","w") as fp:
        json.dump(pf,fp)
        
def main():
    load_resources()
    load_cpnet()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input data dir. Should contain the concepts (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default="../datasets/arc_data/",
        type=str,
        required=True,
        help="The output data dir.",
    )
    parser.add_argument( "--typ", default="train", type=str,required=True, help="Datasets type i.e., [train,easy,test]")
    parser.add_argument( "--start", default=0, type=int, help="starting index of arc question")
    parser.add_argument("--end",default=250,type=int, help="last index of arc question")
    args = parser.parse_args()
    with open(args.data_dir,'r') as f:
        data = json.load(f)
    process_ob(args,data)

if __name__ == '__main__':
    main()

    