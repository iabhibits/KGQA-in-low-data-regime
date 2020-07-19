import json
import numpy as np
import string
import re
import os
import pickle
#import load_kg_emb

class KG :

	def __init__(self, path_to_kg=None) :
		self.path_to_kg = path_to_kg
		self.read_entity_and_relation_list()
		with open('emb_dump.dp','rb') as handle :
			self.kg_embeddings = pickle.load(handle)
		#self.kg_embeddings = load_kg_emb.get_kg_emb()
		self.num_relations,self.relation_dim = self.kg_embeddings['rel_matrices'].shape
		self.num_entities, self.entity_dim = self.kg_embeddings['ent_embeddings'].shape
		x = np.zeros(self.entity_dim)
		self.kg_embeddings['ent_embeddings'] = np.vstack([self.kg_embeddings['ent_embeddings'], x])

	def read_entity_and_relation_list(self) :
		self.entity2id = dict()
		with open(self.path_to_kg+'/entity2id.txt','r') as f :
		    num_ent = int(f.readline()) 
		    for i in range(num_ent) :
		        line = f.readline()
		        line = line.split()
		        self.entity2id[line[0][2:-1]] = int(line[1])
		self.relation2id = dict()
		with open(self.path_to_kg+'/relation2id.txt','r') as f :
		    num_rel = int(f.readline()) 
		    for i in range(num_rel) :
		        line = f.readline()
		        line = line.split()
		        self.relation2id[line[0][2:-1]] = int(line[1])                

if __name__ == '__main__' :
	kg = KG('/home/abhishek/QA/Bert/')
	print(type(kg.kg_embeddings['rel_matrices']))
	print(kg.kg_embeddings)