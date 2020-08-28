import json 
from generate_vocab import UNK, PAD

def loadData(f_pth="./data/toy-data.txt"):
	with open(f_pth, 'r') as f:
		# Each element of X is [ast, index]
		# where index is the starting position of the unseen nodes
		# See separate_dps in utils.py for more details
		X = [json.loads(line) for line in f]
	y = [x[0].pop() for x in X]
	return X, y

class Tokeniser():
	def __init__(self, vocab_filepath, max_len=100000):
		vocab = json.load(open(vocab_filepath))
		self.vocab = {vocab[i]: i for i in range(len(vocab))}
		self.vocab.update({i: vocab[i] for i in range(len(vocab))})
		self.max_len = max_len
		self.UNK, self.PAD = UNK, PAD
	
	def encode(self, sequence):
		return [self.vocab.get(s, self.UNK) for s in sequence] + \
				[self.PAD for _ in range(self.max_len-len(sequence))]
	
	def decode(self, q):
		return self.vocab[q]
	


def prepBatchData(X, y):
	# Function for tokenising elements of X and y
	pass