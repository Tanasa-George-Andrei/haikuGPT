import numpy as np
import torch 
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from torchtext.data import get_tokenizer
import pickle
import time
from os.path import exists
import random

batch_size = 64
block_size = 32
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 300
n_head = 6
n_layer = 6
dropout = 0.2
model_file_name = '_v3'
load_model_from_file = True
train_model = True
save_model_to_file = True
number_to_generate = 200
max_haiku_size = 50

tokenizer = get_tokenizer("basic_english")


with open('haiku_cleaned_dataset.txt', 'r') as dataset_file:
    haiku_text = [tokenizer(text) for text in dataset_file.read().split('\n')]
    random.shuffle(haiku_text)

tokens = list(set([item for sublist in haiku_text for item in sublist]))
tokens = sorted(tokens)

def pad_list(l):
    while len(l) < block_size + 1:
        l.append('<eos>')

for l in haiku_text:
    pad_list(l)

vocab_size = len(tokens)
stoi = { t:i for i,t in enumerate(tokens) }
itos = { i:t for i,t in enumerate(tokens) }

def encodet(t):
    return stoi['<err>'] if t not in stoi else stoi[t]

def encodel(l):
    return [encodet(t) for t in l]

def decode(l, format=True):
    result = ' '.join([itos[i] for i in l])
    if format:
        result = result.split('<sos> ')[1]
        result = result.replace(' . ', '. ').replace(' , ', ', ').replace(' ? ', '? ').replace(' ! ', '! ').replace(' <eor> ','\n').replace(' <eos>', '\n\n').replace('  ',' ').replace(' \' ', '\'')
    
    return result


print(f"Number of tokens: {vocab_size}")

glove = pd.read_csv('glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}

if exists(f'embedding{model_file_name}.pkl'):
    with open(f'embedding{model_file_name}.pkl', 'rb') as f:
        embedding_matrix = pickle.load(f)
    print('Embedding loaded successfully')
else:
    def create_embedding_matrix(embedding_dict):
        embedding_matrix=np.zeros((vocab_size,n_embd))
    
        for word in tokens:
            if word in embedding_dict:
                embedding_matrix[encodet(word)] = embedding_dict[word]
        return embedding_matrix

    embedding_matrix = torch.tensor(create_embedding_matrix(glove_embedding), dtype=torch.float32)

    with open(f'embedding{model_file_name}.pkl', 'wb') as f:
        pickle.dump(embedding_matrix,f)

dataset = [torch.tensor(encodel(l), dtype=torch.long) for l in haiku_text]

train_data_size = int(0.8*len(dataset))
train_data = dataset[:train_data_size]
val_data = dataset[train_data_size:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.stack([data[i][:block_size] for i in ix])
    y = torch.stack([data[i][1:block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


    
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):


    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):


    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):


    def __init__(self, n_embd, n_head):

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, embedding_matrix):
        super().__init__()

        self.token_embedding_table = nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx):
        idx[0][0] = stoi['<sos>']
        last_token = stoi['<sos>']
        for _ in range(max_haiku_size):
            idx_cond = idx[:, -block_size:]
            logits, loss = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx[0][-1] == stoi['<eos>']:
                break
        return idx

model = GPTLanguageModel(embedding_matrix)
if load_model_from_file and exists(f'model{model_file_name}.pkl'):
    print("Loading model parameters...")
    with open(f'model{model_file_name}.pkl','rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

if train_model:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    eval_interval_time = time.time()
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"Eval time: {(time.time() - eval_interval_time):.2f} s, Remaining Time: {((max_iters - iter)*(time.time() - eval_interval_time)/eval_interval):.2f}")
            losses = estimate_loss()
            print(f"Step {iter}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
            eval_interval_time = time.time()
            if save_model_to_file:   
                with open(f'model{model_file_name}.pkl', 'wb') as f:
                    pickle.dump(model,f)
                    print("Model saved successfully")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
     
if save_model_to_file:   
    with open(f'model{model_file_name}.pkl', 'wb') as f:
        pickle.dump(model,f)
    print("Model saved successfully")

with open('output.txt','w') as f:
    for _ in range(number_to_generate):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        f.write(decode(m.generate(context)[0].tolist(), True))