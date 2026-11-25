"""GRPO: Group Relative Policy Optimization (DeepSeekMath, 2024).

One training step:
  1. Rollouts: for each of B prompts, sample a group of G answers.
  2. Rewards: score every answer.
  3. Advantages: within each group, advantage = (reward - mean) / std.
     Better than its siblings -> positive, worse -> negative. A group where
     every answer scored the same has zero advantage everywhere and teaches
     nothing (we skip it and count it: "groups with signal").
  4. Loss, per generated token:  -(advantage * ratio)  +  beta * KL(policy || reference)
     where ratio = P_new(token) / P_old(token). Minimising it raises the
     probability of tokens in above-average answers and lowers it in
     below-average ones; the KL term is the leash to the starting model.
  5. One optimizer step.

About the ratio and clipping: PPO-style training takes several gradient steps
on the same batch of rollouts. The ratio says how far the policy has moved
since the rollouts were sampled, and clipping it to [1-eps, 1+eps] stops one
batch from moving any token's probability too far. We take ONE step per
batch, so the ratio is exactly 1 and clipping never triggers; it's kept so
the formula matches the paper and so multi-step updates are a flag away.
"""

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .model import chat_prompt, completion_logprobs, sample_group


@dataclass
class Config:
    groups_per_step: int = 4  # B: prompts per step
    G: int = 8  # answers per prompt
    max_new_tokens: int = 200
    temp: float = 1.0
    beta: float = 0.04  # KL leash strength (DeepSeekMath used 0.04)
    clip_eps: float = 0.2


def grpo_loss(policy, prompt, completions, adv, ref_logp, beta, eps):
    logp, mask = completion_logprobs(policy, prompt, completions)
    old_logp = mx.stop_gradient(logp)  # policy at sampling time == now (one step per batch)
    ratio = mx.exp(logp - old_logp)
    a = adv[:, None]
    surrogate = mx.minimum(ratio * a, mx.clip(ratio, 1 - eps, 1 + eps) * a)
    # Per-token KL(policy || reference), the "k3" estimator: exp(d) - d - 1 with
    # d = log P_ref - log P_policy. Always >= 0, and 0 when they agree.
    d = ref_logp - logp
    kl = mx.exp(d) - d - 1
    per_token = -(surrogate - beta * kl)
    n = mask.sum()
    return (per_token * mask).sum() / n, (kl * mask).sum() / n


def train_step(policy, ref, tok, optimizer, tasks, reward_batch, cfg: Config):
    t0 = time.time()

    # 1-3. Rollouts, rewards, advantages.
    groups = []
    for task in tasks:
        g = sample_group(policy, tok, chat_prompt(tok, task["messages"]), cfg.G, cfg.max_new_tokens, cfg.temp)
        scores = reward_batch(g.texts, task)  # e.g. run every answer's tests in the sandbox
        r = mx.array([s["reward"] for s in scores])
        std = r.std().item()
        adv = (r - r.mean()) / (std + 1e-4)
        groups.append((task, g, scores, adv, std))
    t_rollout = time.time()

    # 4-5. Loss and one optimizer step, accumulating gradients over the groups
    # that have signal (one group at a time keeps memory low).
    loss_and_grad = nn.value_and_grad(policy, grpo_loss)
    grads, losses, kls = None, [], []
    for task, g, scores, adv, std in groups:
        if std == 0:
            continue
        ref_logp, _ = completion_logprobs(ref, g.prompt, g.completions)
        (loss, kl), gr = loss_and_grad(policy, g.prompt, g.completions, adv, mx.stop_gradient(ref_logp), cfg.beta, cfg.clip_eps)
        grads = gr if grads is None else tree_map(mx.add, grads, gr)
        mx.eval(grads, loss, kl)
        losses.append(loss.item())
        kls.append(kl.item())
    if grads is not None:
        k = len(losses)
        grads = tree_map(lambda x: x / k, grads)
        optimizer.update(policy, grads)
        mx.eval(policy.trainable_parameters(), optimizer.state)
    t_train = time.time()

    all_scores = [s for _, _, scores, _, _ in groups for s in scores]
    n = len(all_scores)
    metrics = {
        "reward": sum(s["reward"] for s in all_scores) / n,
        "groups_with_signal": sum(1 for *_, std in groups if std > 0),
        "groups": len(groups),
        "completion_tokens": sum(len(c) for _, g, *_ in groups for c in g.completions) / n,
        "kl": sum(kls) / len(kls) if kls else 0.0,
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "rollout_s": t_rollout - t0,
        "train_s": t_train - t_rollout,
    }
    for key in all_scores[0]:
        if key != "reward" and isinstance(all_scores[0][key], bool):
            metrics[key] = sum(s[key] for s in all_scores) / n
    return metrics, groups


def evaluate(policy, tok, tasks, reward_batch, max_new_tokens):
    """Greedy (most likely token every time) on a fixed task set: a steadier
    view of progress than the noisy training reward."""
    total = {}
    for task in tasks:
        g = sample_group(policy, tok, chat_prompt(tok, task["messages"]), 1, max_new_tokens, temp=0.0)
        for k, v in reward_batch(g.texts, task)[0].items():
            total[k] = total.get(k, 0) + float(v)
    return {k: v / len(tasks) for k, v in total.items()}
