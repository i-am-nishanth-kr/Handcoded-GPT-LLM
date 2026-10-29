import os
import math
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class GPT2Config:
    def __init__(self):
        self.vocab_size = 50257
        self.embed_dim = 768
        self.seq_len = 1024 # this was called as block_size in karpathy code.
        self.n_blocks = 12
        self.n_heads = 12 
        
    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        c = cls()
        c.__dict__.update(d)
        return c

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.linear_2 = nn.Linear(4 * config.embed_dim, config.embed_dim)
        # no input params for GELU
        self.gelu = nn.GELU()
        
    def forward(self, inputs):
        x = self.linear_1(inputs)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.head_dim = self.embed_dim // self.n_heads # 32 dims per head
        self.q_lin = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_lin = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_lin = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        # torch.ones of full context window should be taken and lower triangle needs to be taken
        self.register_buffer('casual_mask', torch.tril(torch.ones(config.seq_len, config.seq_len)))
        
    
    def forward(self, x):
        
        B, T, C = x.shape
        
        wq = self.q_lin(x)
        wk = self.k_lin(x)
        wv = self.v_lin(x)
        # QKV matrices to be splitted into 4 dimensions
        wq = wq.view(B, T, self.n_heads, self.head_dim)
        wk = wk.view(B, T, self.n_heads, self.head_dim)
        wv = wv.view(B, T, self.n_heads, self.head_dim)
        
        # now for a batch if inputs, each input tokens, the embedding is split for number of heads.
        # we need to take this for matrix multiplication in attention. to make that efficient, we can group the embedding per per sentence to a head to embedding of the entire batch per head 
        # swap dims 1 and 2 to get 2D matrix per head.
        
        wq = wq.transpose(1,2)
        wk = wk.transpose(1,2)
        wv = wv.transpose(1,2)
        
        # TODO -> does the below work for multi head attention?
        op_1 = wq @ wk.transpose(-2, -1)
        x = op_1 / (self.head_dim ** 0.5) # square root of head dims
        
        mask = self.casual_mask[:T, :T]
        x.masked_fill(mask==0, float("-inf"))
        
        x = F.softmax(x, dim = -1)
        x = self.dropout(x)
        x = x @ wv
        # attention scores
        x = x.transpose(1, 2).contiguous().view(B, T, C)

        out = self.linear_out(x)
        return out
        

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ln_f1 = nn.LayerNorm(config.embed_dim)
        self.mha = MHA(config)
        self.ln_f2 = nn.LayerNorm(config.embed_dim)
        
        self.dropout_1 = nn.Dropout(0.1)
        self.mlp = MLP(config)
        self.dropout_2 = nn.Dropout(0.1)

        
    def forward(self, inputs):
        # inputs shape is (B, T, C)
        x = self.ln_f1(inputs)
        x = self.mha(x)
        x = self.dropout_1(x)
        x = x + inputs # residual 1

        op_1 = self.ln_f2(x)
        op_1 = self.mlp(op_1)
        op_1 = self.dropout_2(op_1)
        # residual must add the UNnormalized x, not the layernormed one
        x = x + op_1
        return x
        



class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.seq_len, config.embed_dim)
        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_blocks)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # scale down the layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("linear_out.weight") or name.endswith("mlp.linear_2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_blocks))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        B, T = token_ids.shape
        assert T <= self.config.seq_len, "sequence longer than context window"

        wte = self.token_embedding(token_ids)
        pos = torch.arange(T, device=token_ids.device)
        wpe = self.pos_embedding(pos)
        x = self.dropout(wte + wpe)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                      # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        # call model.eval() before this so dropout is off
        for _ in range(max_new_tokens):
            ctx = token_ids[:, -self.config.seq_len:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-6)   # (B, vocab)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)       # (B, 1)
            token_ids = torch.cat([token_ids, next_id], dim=1)
        return token_ids


# data part
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_data(parquet_path, cache_dir="data_cache", val_fraction=0.05):
    """Tokenize the 'text' column once and cache it as uint16 numpy files."""
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "train.npy")
    val_path = os.path.join(cache_dir, "val.npy")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("loading cached tokens from", cache_dir)
        return np.load(train_path, mmap_mode="r"), np.load(val_path, mmap_mode="r")

    import pandas as pd
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")   # same 50257 vocab as the model

    df = pd.read_parquet(parquet_path, columns=["text"])
    texts = df["text"].dropna().astype(str).tolist()
    print(f"{len(texts)} stories")

    # split by story (not by token) so val stories are never seen in training
    rng = np.random.default_rng(1337)
    perm = rng.permutation(len(texts))
    n_val = max(1, int(len(texts) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def encode(indices):
        chunks = []
        for start in range(0, len(indices), 1000):
            batch = [texts[i] for i in indices[start:start + 1000]]
            for toks in enc.encode_ordinary_batch(batch, num_threads=8):
                toks.append(enc.eot_token)    # <|endoftext|> between stories
                chunks.append(np.array(toks, dtype=np.uint16))
        return np.concatenate(chunks)

    train_data, val_data = encode(train_idx), encode(val_idx)
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    print(f"train tokens: {len(train_data):,}  val tokens: {len(val_data):,}")
    return train_data, val_data


def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# Train / eval
def get_lr(step, max_lr, warmup, max_steps):
    min_lr = max_lr * 0.1
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device, amp_ctx):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(args.eval_iters)
        for i in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, model.config.seq_len, device)
            with amp_ctx:
                logits = model(x)
            losses[i] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, model, optimizer, step, best_val, config):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": config.to_dict(),
    }, path)


def train(args):
    device = get_device()
    print("device:", device)
    os.makedirs(args.out_dir, exist_ok=True)
    train_data, val_data = prepare_data(args.data)

    config = GPT2Config()
    config.n_blocks = args.n_blocks
    start_step, best_val = 0, float("inf")
    ckpt = None
    latest_path = os.path.join(args.out_dir, "latest.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        config = GPT2Config.from_dict(ckpt["config"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"resuming from step {start_step}")

    model = GPT2(config).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # weight decay only on matrices (not biases / layernorm)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        pass

    model.train()
    t0 = time.time()
    for step in range(start_step, args.max_steps + 1):
        lr = get_lr(step, args.lr, args.warmup, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- evaluation + checkpoint ----
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args, device, amp_ctx)
            print(f"[eval] step {step}: train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| val ppl {math.exp(losses['val']):.1f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(os.path.join(args.out_dir, "best.pt"),
                                model, optimizer, step, best_val, config)
                print("  saved best.pt")
            save_checkpoint(latest_path, model, optimizer, step, best_val, config)

            if enc is not None and step > 0:
                model.eval()
                prompt = torch.tensor([enc.encode_ordinary("Once upon a time")], device=device)
                sample = model.generate(prompt, 60, temperature=0.8, top_k=50)
                print("  sample:", enc.decode(sample[0].tolist()).replace("\n", " "))
                model.train()

        if step == args.max_steps:
            break

        # ---- one training step ----
        x, y = get_batch(train_data, args.batch_size, config.seq_len, device)
        with amp_ctx:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s/50 steps")

# Generation from a saved checkpoint
def generate_text(args):
    import tiktoken
    torch.manual_seed(42)
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPT2Config.from_dict(ckpt["config"])
    model = GPT2(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ckpt['step']}, best val loss {ckpt['best_val']:.4f})")

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode_ordinary(args.prompt)], device=device)
    out = model.generate(ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print("-" * 60)
    print(enc.decode(out[0].tolist()))

# dummy sanity test
def run_tests():
    torch.manual_seed(0)
    config = GPT2Config()
    model = GPT2(config)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    tokens = torch.randint(0, config.vocab_size, (2, 10))
    print("logits shape:", model(tokens).shape)

    model.eval()
    a = torch.randint(0, config.vocab_size, (1, 10))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % config.vocab_size
    with torch.no_grad():
        la, lb = model(a), model(b)
    print("positions 0-6 unchanged:", torch.allclose(la[:, :7], lb[:, :7], atol=1e-5))
    print("position 7 changed:     ", not torch.allclose(la[:, 7], lb[:, 7], atol=1e-5))

    model.train()
    x, y = tokens[:, :-1], tokens[:, 1:]
    loss = F.cross_entropy(model(x).reshape(-1, config.vocab_size), y.reshape(-1))
    print(f"initial loss: {loss.item():.3f} (expected ~ {math.log(config.vocab_size):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--data", default="60k_stories.parquet")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--resume", action="store_true")

    g = sub.add_parser("generate")
    g.add_argument("--ckpt", default="checkpoints/best.pt")
    g.add_argument("--prompt", default="Once upon a time")
    g.add_argument("--max_new_tokens", type=int, default=150)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_k", type=int, default=50)

    sub.add_parser("test")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "generate":
        generate_text(args)
    else:
        run_tests()
