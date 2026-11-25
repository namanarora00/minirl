"""How good is the untrained model? Sample G answers for N tasks and report
how many groups have signal (some right, some wrong): only those teach GRPO."""
import random, sys, time
import mlx.core as mx
from minirl.model import load_policy, chat_prompt, sample_group
from minirl.tasks import arith

N, G = int(sys.argv[1]) if len(sys.argv) > 1 else 10, 8
mx.random.seed(0)
model, tok = load_policy()
rng = random.Random(0)
stats = {"correct": 0, "formatted": 0, "signal_groups": 0, "tokens": 0}
t0 = time.time()
for i in range(N):
    task = arith.make_task(rng)
    g = sample_group(model, tok, chat_prompt(tok, task["messages"]), G, max_new_tokens=200)
    rs = [arith.reward(t, task) for t in g.texts]
    c = sum(r["correct"] for r in rs)
    stats["correct"] += c
    stats["formatted"] += sum(r["formatted"] for r in rs)
    stats["signal_groups"] += 0 < c < G
    stats["tokens"] += sum(len(x) for x in g.completions)
    print(f"task {i}: {task['messages'][0]['content'].splitlines()[0]}  correct {c}/{G}  formatted {sum(r['formatted'] for r in rs)}/{G}", flush=True)
dt = time.time() - t0
print(f"\naccuracy {stats['correct']/(N*G):.0%}  format {stats['formatted']/(N*G):.0%}  groups with signal {stats['signal_groups']}/{N}  "
      f"{stats['tokens']} tokens in {dt:.0f}s = {stats['tokens']/dt:.0f} tok/s")
print("--- example completion ---\n" + g.texts[0])
