"""Sanity-check task: two-digit multiplication.

Why arithmetic first: the reward is a few lines of Python, no sandbox, so if
training doesn't improve it, the bug is in the trainer, not the environment.

The reward is strict on purpose: the answer only counts on a final line
"ANSWER: <number>". The model also has to learn the format, which RL picks up
fast, and which shows a key RL lesson: the model learns whatever the reward
function actually checks, not what you meant.
"""

import random
import re

PROMPT = "Compute {a} * {b}. Work it out briefly, then end with a final line of the form:\nANSWER: <number>"


def make_task(rng: random.Random, hard_fraction=0.5):
    """Difficulty mix. The untrained model solves ~5% of 2-digit x 2-digit,
    which leaves most GRPO groups all-wrong: zero advantage, nothing learned.
    2-digit x 1-digit lands in the "sometimes right" zone where learning
    happens; keeping some hard ones gives room to improve."""
    a = rng.randint(10, 99)
    b = rng.randint(10, 99) if rng.random() < hard_fraction else rng.randint(2, 9)
    return {"messages": [{"role": "user", "content": PROMPT.format(a=a, b=b)}], "answer": a * b, "hard": b >= 10}


ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d,]+)\s*$")


def reward(text: str, task) -> dict:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    m = ANSWER_RE.search(lines[-1]) if lines else None
    formatted = m is not None
    correct = formatted and int(m.group(1).replace(",", "")) == task["answer"]
    return {"reward": 1.0 if correct else 0.0, "formatted": formatted, "correct": correct}


MAX_NEW = 200


def reward_batch(texts, task):
    return [reward(t, task) for t in texts]
