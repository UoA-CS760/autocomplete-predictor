from model import TransformerModel
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from data_prep import Tokeniser, Dataset
np.random.seed(0)

n_ctx = 1000
vocab_size = 50002 #100000
def buildModel():
	layer_norm_epsilon = 1e-6
	n_head = 2 #6
	n_embd = 300
	n_layer = 2 #6
	loss_fn = CrossEntropyLoss
	return TransformerModel(vocab_size, loss_fn, n_layer, n_embd, n_ctx, n_head, layer_norm_epsilon)

t = buildModel()
t.train()
t.zero_grad()

tokeniser = Tokeniser('vocab50k.json', max_seq_len=n_ctx)
data = Dataset()

batch_size=20
# for _ in range(training_steps):
X, y = data.getBatch(batch_size=batch_size)
X, y = tokeniser.encode_batch(X, y, tensor=True)
y_hot = torch.zeros(batch_size, vocab_size, dtype=torch.bool)
for i, j in enumerate(y):
	y_hot[i,j] = True

output = t(X, y)
print(output.shape)