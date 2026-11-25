"""The model side of RL: load a policy, sample answers from it, and score how
likely it finds a given answer (log-probabilities).

Two copies of the same model are in play:
  policy     the one we train. Its base weights stay frozen; only small LoRA
             adapters (low-rank matrices added to each linear layer) learn.
             Cheap on memory, and enough for RL, which nudges behaviour more
             than it teaches new knowledge.
  reference  the untouched starting model. GRPO's KL "leash" compares the
             policy's probabilities against it, so the policy can't drift
             into nonsense that happens to score well.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.utils import linear_to_lora_layers

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def load_policy(rank=16, scale=2.0):
    model, tok = load(MODEL)
    model.freeze()
    n_layers = len(model.model.layers)
    linear_to_lora_layers(model, n_layers, {"rank": rank, "scale": scale, "dropout": 0.0, "keys": LORA_KEYS})
    return model, tok


def load_reference():
    model, _ = load(MODEL)
    model.freeze()
    return model


def chat_prompt(tok, messages):
    """Token ids of a chat, ending where the assistant should start writing."""
    return tok.apply_chat_template(messages, add_generation_prompt=True)


@dataclass
class Group:
    prompt: list  # token ids
    completions: list  # G lists of token ids, each ending at (and including) EOS if it stopped
    texts: list  # decoded completions


def sample_group(model, tok, prompt_ids, G, max_new_tokens=200, temp=1.0):
    """G answers to one prompt, generated together as a batch of G.

    How generation works: one forward pass over the prompt fills a KV cache
    (the attention keys/values of every token so far). Then each new token
    needs just one more forward pass over that single token, reusing the
    cache. Sampling at temperature 1.0 draws from the model's own
    distribution: RL needs varied attempts, not the single most likely one.
    """
    eos = set(tok.eos_token_ids)
    cache = make_prompt_cache(model)
    x = mx.array([prompt_ids] * G)
    logits = model(x, cache=cache)[:, -1, :]
    out = []
    done = [False] * G
    for _ in range(max_new_tokens):
        if temp == 0:
            tok_ids = mx.argmax(logits, axis=-1)  # greedy: for evaluation
        else:
            tok_ids = mx.random.categorical(logits * (1.0 / temp))
        mx.eval(tok_ids)
        ids = tok_ids.tolist()
        out.append(ids)
        for i, t in enumerate(ids):
            if t in eos:
                done[i] = True
        if all(done):
            break
        logits = model(tok_ids[:, None], cache=cache)[:, -1, :]

    completions = []
    for i in range(G):
        seq = []
        for step in out:
            seq.append(step[i])
            if step[i] in eos:
                break
        completions.append(seq)
    texts = [tok.decode([t for t in c if t not in eos]) for c in completions]
    return Group(prompt=list(prompt_ids), completions=completions, texts=texts)


def completion_logprobs(model, prompt, completions):
    """log P(token) for every completion token, under `model`.

    Runs prompt+completion through the model once and reads off the
    probability it gave each token that was actually generated. Returns
    (logp [G, T], mask [G, T]); completions are right-padded to length T
    and mask marks real tokens.

    Only completion positions go through the final vocabulary projection:
    the vocabulary is ~152k entries, so projecting every prompt position too
    would cost gigabytes for nothing.
    """
    P = len(prompt)
    T = max(len(c) for c in completions)
    pad = 0
    ids = mx.array([prompt + c + [pad] * (T - len(c)) for c in completions])
    mask = mx.array([[1.0] * len(c) + [0.0] * (T - len(c)) for c in completions])

    hidden = model.model(ids[:, :-1])  # [G, P+T-1, H]: position j predicts token j+1
    hidden = hidden[:, P - 1 :, :]  # the positions that predict completion tokens
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
    else:
        logits = model.lm_head(hidden)
    logits = logits.astype(mx.float32)
    logp_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    targets = ids[:, P:]
    logp = mx.take_along_axis(logp_all, targets[..., None], axis=-1)[..., 0]
    return logp, mask
