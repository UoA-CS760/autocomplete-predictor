import json 
from generate_vocab import UNK, PAD
import torch

###################################################
# See train.py for example usage of these classes #
###################################################

class Tokeniser(object):
	def __init__(self, vocab_filepath="vocab.json", vocab_size=100000, max_seq_len=1000):
		vocab = json.load(open(vocab_filepath))
		vocab_size = min(vocab_size-2, len(vocab))
		self.vocab = {vocab[i]: i for i in range(vocab_size)}
		self.vocab.update({i: vocab[i] for i in range(vocab_size)})
		self.vocab.update({vocab_size-1: UNK, vocab_size: PAD})
		self.vocab.update({UNK: vocab_size-1, PAD: vocab_size})
		self.max_seq_len = max_seq_len
		self.unk, self.pad = self.vocab[UNK], self.vocab[PAD]
	
	def encode_value(self, val):
		return self.vocab.get(val, self.unk)

	def encode_seq(self, sequence):
		return [self.vocab.get(s, self.unk) for s in sequence] + \
				[self.pad for _ in range(self.max_seq_len-len(sequence))]
	
	def decode_seq(self, seq):
		return [self.vocab[s] for s in seq]
	
	def encode_batch(self, X, Y, tensor=True):
		encoded_X = [self.encode_seq(x) for x in X]
		encoded_Y = [self.encode_value(y) for y in Y]
		if tensor:
			return torch.tensor(encoded_X, dtype=torch.long), torch.tensor(encoded_Y, dtype=torch.long) 
		return encoded_X, encoded_Y
	
class Dataset(object):
	def __init__(self, f_pth="./data/toy-data.txt"):
		self.loadData(f_pth)
		# Keeps track of place in the dataset
		self.batchIndex = 0

	def loadData(self, f_pth):
		with open(f_pth, 'r') as f:
			# Each element of X is [ast, index]
			# where index is the starting position of the unseen nodes
			# See separate_dps in utils.py for more details
			self.X = [json.loads(line) for line in f]
			self.y = [x[0][-1] for x in self.X]
			
	def getBatch(self, batch_size):
		# Function for tokenising elements of X and y
		BI = self.batchIndex
		x_batch = [x[0][:-1] for x in self.X[BI:BI+batch_size]]
		y_batch = self.y[BI:BI+batch_size]
		self.batchIndex += batch_size
		return x_batch, y_batch