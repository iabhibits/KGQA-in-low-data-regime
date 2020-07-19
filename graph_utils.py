import re
import ftfy
import json
import spacy
from tqdm import tqdm
import string
from kg_loader import KG

class GraphEncoder(object) :
    """
    a wrapper for encoding a  graph sequence
    """
    def __init__(self, kg) :
        self.ent_embeddings = kg.kg_embeddings['ent_embeddings']
        self.rel_matrices =  kg.kg_embeddings['rel_matrices']
        self.num_relations = kg.num_relations
        self.num_entities = kg.num_entities
        self.relation_dim =  kg.relation_dim
        self.entity_dim = kg.entity_dim
        self.entity2id = kg.entity2id
        self.relation2id = kg.relation2id               
        self.translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
       
    def encode(self, text) :
        temp = []
        for line in text :
            temp2 = []
            s = set()
            for c in line.translate(self.translator).split() :
                if c.lower() in self.entity2id.keys() and c.lower() not in s :
                    temp2.append(self.entity2id[c.lower()])
                    s.add(c.lower())
            temp.append(temp2)
        return temp 

if __name__ == '__main__' :

    kg = KG('./data/conceptnet')
    print(kg.entity2id['jamjar'])
    ge = GraphEncoder(kg)
    text = ["This is America","My name is Khan."," Are you   mad?"]
    en = ge.encode(text)
    print(ge.ent_embeddings[en[0][0]])
    print(en)

