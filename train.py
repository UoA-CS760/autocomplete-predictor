from model import TransformerModel
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from data_prep import Tokeniser, Dataset
np.random.seed(0)

n_ctx = 1000
def buildModel():
	layer_norm_epsilon = 1e-6
	n_head = 2 #6
	n_embd = 300
	n_layer = 2 #6
	loss_fn = CrossEntropyLoss
	vocab_size = 50002 #100000
	return TransformerModel(vocab_size, loss_fn, n_layer, n_embd, n_ctx, n_head, layer_norm_epsilon)

t = buildModel()
t.train()
t.zero_grad()

tokeniser = Tokeniser('vocab50k.json', max_len=n_ctx)
data = Dataset()

X, y = data.getBatch(batch_size=5)
X, y = tokeniser.encode_batch(X, y, tensor=True)

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

output = t(X, y)
print(output.size())