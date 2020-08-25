import json 
import pickle
from transformers import PreTrainedTokenizer


def loadData(f_pth="./data/toy-data.txt"):
	with open(f_pth, 'r') as f:
		# Each element of X is [ast, index]
		# where index is the starting position of the unseen nodes
		# See separate_dps in utils.py for more details
		X = [json.loads(line) for line in f]
	y = [x[0].pop() for x in X]
	return X, y



def prepBatchData(X, y):
	# Function for tokenising elements of X and y
	pass