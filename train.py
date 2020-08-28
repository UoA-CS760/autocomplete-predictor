from model import TransformerModel
from torch.nn import CrossEntropyLoss
import torch
import numpy as np
from data_prep import Tokeniser
np.random.seed(0)

layer_norm_epsilon = 1e-6
n_head = 2 #6
n_ctx = 1000
n_embd = 300
n_layer = 2 #6
loss_fn = CrossEntropyLoss
vocab_size = 50000 #100000
criterion = CrossEntropyLoss
t = TransformerModel(vocab_size, loss_fn, n_layer, n_embd, n_ctx, n_head, layer_norm_epsilon)
t.train()
tokeniser = Tokeniser('vocab50k.json', max_len=vocab_size)
X = np.round(np.random.rand(10,1000)*1000)
y = np.round(np.random.rand(10)*1000)
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

output = t(X, y)
print(output.size())