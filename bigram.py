import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import os

# hyper params
batch_size = 32 # how many independent sequences to process in parallel
block_size = 128 # max context length for prediction
max_iters = 5000
eval_interval = 200
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 4
n_layer = 4
n_embed = n_head * batch_size
dropout = 0.2
save_file = "model.pt"

# torch.manual_seed(23489)

# todo: get a proper input file
# curl -o input.txt https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
# curl -o input.txt https://norvig.com/big.txt
filename = 'input.txt'

with open(filename, 'r', encoding='utf-8') as f:
  text = f.read()

print("Length of text: ", len(text))
print(text[:1000])

# all unique chars in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print("size: ", vocab_size)

# create mapping from chars to int
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("I love you"))
# print(decode(encode("I love you")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# Setting up train and validation sets
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# print("sample train data encoded, one block:", train_data[:block_size+1])

# illustrating training on a block
# we take a sequence of chars (len 1 to block_size), and see what appears next
# x = train_data[:block_size]
# y = train_data[1:1+block_size]
# for t in range(block_size):
#   context = x[:t+1]
#   target = y[t]
#   print(f"When input is {context} then the target is {target}")

def time_str():
  return datetime.now().strftime('%H:%M:%S')

# data loading
def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

# here's what gets fed into the neural network
xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
# print('====')

# illustrating the training again
# for b in range(batch_size):
#   for t in range(block_size):
#     context = xb[b, :t+1]
#     target = yb[b, t]
#     print(f"When input is {context} then the target is {target}")

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval() # set model to eval mode
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train() # revert model back to train mode
  return out

class Head(nn.Module):
  """ one head of self-attention """
  def __init__(me, head_size):
    super().__init__()
    me.key = nn.Linear(n_embed, head_size, bias=False)
    me.query = nn.Linear(n_embed, head_size, bias=False)
    me.value = nn.Linear(n_embed, head_size, bias=False)
    me.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    me.dropout = nn.Dropout(dropout)

  def forward(me, x):
    B,T,C = x.shape
    k = me.key(x) # (B,T,C)
    q = me.query(x) # (B,T,C)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
    wei = wei.masked_fill(me.tril[:T,:T]==0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    wei = me.dropout(wei)
    # perform the weighted aggregation of the values
    v = me.value(x) # (B,T,C)
    out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
    return out

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention """
  def __init__(me, num_heads, head_size):
    super().__init__()
    me.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    me.proj = nn.Linear(n_embed, n_embed)
    me.dropout = nn.Dropout(dropout)

  def forward(me, x):
    out = torch.cat([h(x) for h in me.heads], dim=-1)
    out = me.proj(out)
    out = me.dropout(out)
    return out

class FeedForward(nn.Module):
  """ a simple linear layer followed by a ReLU non-linearity """
  def __init__(me, n_embed):
    super().__init__()
    me.net = nn.Sequential(
      nn.Linear(n_embed, n_embed * 4),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout),
    )

  def forward(me, x):
    return me.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation """
  def __init__(me, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    me.sa = MultiHeadAttention(n_head, head_size)
    me.ffwd = FeedForward(n_embed)
    me.ln1 = nn.LayerNorm(n_embed)
    me.ln2 = nn.LayerNorm(n_embed)

  def forward(me, x):
    # contrary to the original Attention Is All You Need paper,
    # here the normalization is done before the transformation
    x = x + me.sa(me.ln1(x))
    x = x + me.ffwd(me.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(me):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    me.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    me.position_embedding_table = nn.Embedding(block_size, n_embed)
    # var 1: using single head
    # me.sa_head = Head(n_embed) # one head self attention
    # var 2: using multiple heads
    # me.sa_heads = MultiHeadAttention(4, n_embed // 4) # 4 heads of 8-dimensional self-attention
    # me.ffwd = FeedForward(n_embed)
    # var 3: using blocks of multiple heads
    me.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
    me.ln_f = nn.LayerNorm(n_embed) # final layer norm
    me.lm_head = nn.Linear(n_embed, vocab_size) # language model

  def forward(me, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B,T) tensor of integers
    tok_embed = me.token_embedding_table(idx) # (B,T,C)
    pos_embed = me.position_embedding_table(torch.arange(T, device=device)) # T,C
    x = tok_embed + pos_embed # (B,T,C)
    # var 1: using single head
    # x = me.sa_head(x) # apply one head of self-attention (B,T,C)
    # var 2: using multiple heads
    # x = me.sa_heads(x) # apply multi head of self-attention (B,T,C)
    # x = me.ffwd(x) # (B,T,C)
    # var 3: using blocks of multiple heads
    x = me.blocks(x) # (B,T,C)
    x = me.ln_f(x) # (B,T,C)
    logits = me.lm_head(x) # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      # reshape the matrix to comply to pytorch's spec
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(me, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to teh last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the prediction
      logits, loss = me(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B,C)
      probs = F.softmax(logits, dim=-1) # (B,C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
    return idx

model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

# try generating, which will output garbage since we haven't trained
# but at least we can see that the plumbings work
gen_result = m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)
print(decode(gen_result[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1.e-3)

def train_model():
  # train the model
  for steps in range(max_iters):
    # print loss at some interval
    if steps % eval_interval == 0:
      losses = estimate_loss()
      print(f"[{time_str()}] step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  print(f"[{time_str()}] FINAL: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

if not os.path.exists(save_file):
    print(f"[{time_str()}] ==== Saved model file '{save_file}' not found, training new on {device} ... ====", flush=True)
    train_model()
    print(f"[{time_str()}] Saving to: {save_file}", flush=True)
    torch.save(m.state_dict(), save_file)
else:
    print(f"[{time_str()}] ==== Saved model file '{save_file}' found, loading ... ====", flush=True)
    m.load_state_dict(torch.load(save_file))

# retry the generation with trained model
n_loop = 5
for i in range(n_loop):
    print(f"[{time_str()}] ==== GENERATED TEXT {i+1} ====", flush=True)
    context = torch.randint(0, vocab_size, size=(1, 1), dtype=torch.long, device=device)
    gen_result = m.generate(context, max_new_tokens=500)
    print(decode(gen_result[0].tolist()), flush=True)
