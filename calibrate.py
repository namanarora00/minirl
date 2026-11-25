"""Difficulty calibration: how often does the current model solve each task?

Samples G answers per task and counts successes. GRPO only learns from tasks
in the band 0 < successes < G: a task the model always solves, or never
solves, gives every answer the same reward, so zero advantage. Prints the
histogram and writes the learnable task ids, to pass to training as
MINIRL_TASK_IDS.

    .venv/bin/python calibrate.py code train
"""
import json, os, sys, time
import mlx.core as mx
from minirl.model import load_policy, chat_prompt, sample_group
from minirl.tasks import code

split = sys.argv[2] if len(sys.argv) > 2 else "train"
G = 8
mx.random.seed(0)
model, tok = load_policy()
tasks = [code.with_messages(t) for t in code.load_tasks() if t["split"] == split]
counts, t0, ntok = {}, time.time(), 0
for i, task in enumerate(tasks, 1):
    g = sample_group(model, tok, chat_prompt(tok, task["messages"]), G, code.MAX_NEW)
    ntok += sum(len(c) for c in g.completions)
    rs = code.reward_batch(g.texts, task)
    counts[task["id"]] = sum(r["correct"] for r in rs)
    print(f"[{i}/{len(tasks)}] {task['id']:<22} {task['mutation']:<16} solved {counts[task['id']]}/{G}  formatted {sum(r['formatted'] for r in rs)}/{G}", flush=True)
hist = {k: sum(1 for c in counts.values() if c == k) for k in range(G + 1)}
learnable = [tid for tid, c in counts.items() if 0 < c < G]
os.makedirs("runs", exist_ok=True)
with open(f"runs/calibration-{split}.json", "w") as f:
    json.dump({"G": G, "counts": counts, "learnable": learnable}, f, indent=1)
print(f"\nsolved-out-of-{G} histogram: {hist}")
print(f"never solved {hist[0]}, always solved {hist[G]}, learnable {len(learnable)} of {len(tasks)}  "
      f"({ntok} tokens in {time.time()-t0:.0f}s = {ntok/(time.time()-t0):.0f} tok/s)")
