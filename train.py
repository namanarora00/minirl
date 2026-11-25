"""Train a model with GRPO.

    .venv/bin/python train.py --task arith --steps 60

Prints one line per step and writes runs/<name>/metrics.jsonl. Every
--eval-every steps it runs a greedy evaluation on a fixed held-out task set.
"""

import argparse
import json
import os
import random
import time

import mlx.core as mx
import mlx.optimizers as optim

from minirl.grpo import Config, evaluate, train_step
from minirl.model import load_policy, load_reference
from minirl.tasks import arith, code

TASKS = {"arith": arith, "code": code}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="arith", choices=TASKS)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max-new", type=int, default=None, help="default: the task's MAX_NEW")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    task_mod = TASKS[args.task]
    name = args.name or f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = f"runs/{name}"
    os.makedirs(out, exist_ok=True)
    mx.random.seed(args.seed)
    train_rng, eval_rng = random.Random(args.seed), random.Random(10_000 + args.seed)
    if args.max_new is None:
        args.max_new = task_mod.MAX_NEW
    if hasattr(task_mod, "eval_tasks"):  # fixed held-out set (code: functions never trained on)
        eval_tasks = task_mod.eval_tasks()
    else:
        eval_tasks = [task_mod.make_task(eval_rng) for _ in range(args.eval_n)]

    policy, tok = load_policy()
    ref = load_reference()
    optimizer = optim.Adam(learning_rate=args.lr)
    cfg = Config(groups_per_step=args.groups, G=args.G, max_new_tokens=args.max_new, beta=args.beta)
    log = open(f"{out}/metrics.jsonl", "w")

    def run_eval(step):
        e = evaluate(policy, tok, eval_tasks, task_mod.reward_batch, args.max_new)
        print(f"  eval @ step {step}: " + "  ".join(f"{k} {v:.0%}" for k, v in e.items()), flush=True)
        log.write(json.dumps({"step": step, "eval": e}) + "\n")
        log.flush()

    run_eval(0)
    for step in range(1, args.steps + 1):
        tasks = [task_mod.make_task(train_rng) for _ in range(args.groups)]
        m, groups = train_step(policy, ref, tok, optimizer, tasks, task_mod.reward_batch, cfg)
        m["step"] = step
        log.write(json.dumps(m) + "\n")
        log.flush()
        extra = "  ".join(f"{k} {m[k]:.0%}" for k in ("correct", "formatted") if k in m)
        print(
            f"step {step:3d}  reward {m['reward']:.2f}  {extra}  signal {m['groups_with_signal']}/{m['groups']}  "
            f"len {m['completion_tokens']:.0f}  kl {m['kl']:.4f}  rollout {m['rollout_s']:.1f}s  train {m['train_s']:.1f}s",
            flush=True,
        )
        if step % args.eval_every == 0:
            run_eval(step)
    # The answer to "what did it learn to write?": the last group's first answer.
    print("--- sample from the final policy ---\n" + groups[-1][1].texts[0])
    adapters = dict(tree for tree in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(policy.trainable_parameters()))
    mx.save_safetensors(f"{out}/adapters.safetensors", adapters)
    print(f"saved {out}/adapters.safetensors and {out}/metrics.jsonl")


if __name__ == "__main__":
    main()
