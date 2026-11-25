"""Bug-fixing tasks: a tiny version of how RL environment companies build
coding tasks (e.g. Abundant's SWE-gen, which reintroduces a real bug into a
repo whose tests pass).

Building a task:
  1. Start from correct code whose tests all pass.
  2. Make one small mechanical change (a "mutation": < to <=, + to -, 0 to 1,
     and to or, in to not in).
  3. Keep it only if (a) the original passes every test and (b) the mutant
     fails at least one VISIBLE test, so the bug is observable from what the
     model is shown.
The model sees the buggy file and the failing visible-test output. The reward
runs the HIDDEN tests (visible + extra edge cases) on its fix, in the sandbox.
Four seed functions are held out entirely, to measure whether training
generalises to functions the model never trained on.
"""

import ast
import copy
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .code_seeds import SEEDS

MAX_NEW = 400
EVAL_SEEDS = {"median", "compress", "transpose", "word_count"}
TASKS_FILE = os.path.join(os.path.dirname(__file__), "code_tasks.json")

RUNNER = '''import sys
mod = __import__(sys.argv[1])
names = sorted((n for n in dir(mod) if n.startswith("test_")), key=lambda n: int(n.split("_")[1]))
passed = 0
for n in names:
    try:
        getattr(mod, n)()
        passed += 1
        print(f"PASS {n}")
    except AssertionError as e:
        print(f"FAIL {n}: {e}")
    except Exception as e:
        print(f"FAIL {n}: {type(e).__name__}: {e}")
print(f"RESULT {passed}/{len(names)}")
'''

PROMPT = """The file solution.py has a bug. Running its tests prints:

```
{output}
```

Fix the bug. Reply with the complete corrected solution.py in a single ```python code block.

```python
{code}
```"""


def test_module(checks):
    lines = ["from solution import *", ""]
    for i, check in enumerate(checks, 1):
        call, want = check.rsplit(" == ", 1)
        lines += [
            f"def test_{i}():",
            f"    got = {call}",
            f"    want = {want}",
            f"    assert got == want, {call!r} + f' returned {{got!r}}, expected {{want!r}}'",
            "",
        ]
    return "\n".join(lines)


# --- mutations ---------------------------------------------------------------

CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In}
BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add}
BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def mutation_sites(tree):
    """Every (node, field, index-or-None, change, description) single-edit site."""
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in CMP_SWAP:
                    sites.append((node, "ops", i, CMP_SWAP[type(op)](), f"{type(op).__name__} -> {CMP_SWAP[type(op)].__name__}"))
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BIN_SWAP:
            sites.append((node, "op", None, BIN_SWAP[type(node.op)](), f"{type(node.op).__name__} -> {BIN_SWAP[type(node.op)].__name__}"))
        elif isinstance(node, ast.BoolOp) and type(node.op) in BOOL_SWAP:
            sites.append((node, "op", None, BOOL_SWAP[type(node.op)](), f"{type(node.op).__name__} -> {BOOL_SWAP[type(node.op)].__name__}"))
        elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in (0, 1, 2):
            for d in (1, -1):
                sites.append((node, "value", None, node.value + d, f"constant {node.value} -> {node.value + d}"))
    return sites


def mutants(code):
    tree = ast.parse(code)
    n = len(mutation_sites(tree))
    seen = set()
    for k in range(n):
        t = copy.deepcopy(tree)
        node, field, idx, new, desc = mutation_sites(t)[k]
        if idx is None:
            setattr(node, field, new)
        else:
            getattr(node, field)[idx] = new
        src = ast.unparse(t)
        if src != code and src not in seen:
            seen.add(src)
            yield src, desc


# --- building and validating tasks -------------------------------------------


def run_tests(pool, code, tests_src, timeout=5):
    _, out = pool.run({"solution.py": code, "tests.py": tests_src, "run_tests.py": RUNNER}, "python run_tests.py tests", timeout=timeout)
    m = re.search(r"RESULT (\d+)/(\d+)", out)
    passed, total = (int(m.group(1)), int(m.group(2))) if m else (0, -1)
    return passed, total, out.strip()


def build_tasks(pool):
    jobs = []
    for seed in SEEDS:
        original = ast.unparse(ast.parse(seed["code"]))
        tv = test_module(seed["visible"])
        th = test_module(seed["visible"] + seed["hidden"])
        p, t, out = run_tests(pool, original, th)
        assert p == t > 0, f"seed {seed['name']} fails its own tests:\n{out}"
        for buggy, desc in mutants(original):
            jobs.append((seed["name"], original, buggy, desc, tv, th))

    def check(job):
        name, original, buggy, desc, tv, th = job
        p, t, out = run_tests(pool, buggy, tv)
        return job, p < t, out

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(check, jobs))
    tasks, per_seed = [], {}
    for (name, original, buggy, desc, tv, th), observable, out in results:
        if not observable:
            continue
        per_seed[name] = per_seed.get(name, 0) + 1
        tasks.append({
            "id": f"{name}-{per_seed[name]}", "seed": name, "split": "eval" if name in EVAL_SEEDS else "train",
            "mutation": desc, "buggy": buggy, "original": original, "visible_output": out,
            "test_visible": tv, "test_hidden": th,
        })
    return tasks, len(jobs)


def load_tasks():
    with open(TASKS_FILE) as f:
        return json.load(f)


def with_messages(task):
    task = dict(task)
    task["messages"] = [{"role": "user", "content": PROMPT.format(output=task["visible_output"], code=task["buggy"])}]
    return task


# --- the interface train.py uses ----------------------------------------------

_pool = None
_train = None


def pool():
    global _pool
    if _pool is None:
        import atexit

        from ..sandbox.docker import DockerPool

        _pool = DockerPool(8)
        atexit.register(_pool.close)
    return _pool


def make_task(rng: random.Random):
    global _train
    if _train is None:
        allowed = os.environ.get("MINIRL_TASK_IDS")  # set by calibration: the learnable band
        _train = [t for t in load_tasks() if t["split"] == "train" and (not allowed or t["id"] in allowed.split(","))]
    return with_messages(rng.choice(_train))


def eval_tasks():
    return [with_messages(t) for t in load_tasks() if t["split"] == "eval"]


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def reward_batch(texts, task):
    def one(text):
        blocks = CODE_RE.findall(text)
        if not blocks:
            return {"reward": 0.0, "formatted": False, "correct": False}
        p, t, _ = run_tests(pool(), blocks[-1], task["test_hidden"])
        correct = t > 0 and p == t
        return {"reward": 1.0 if correct else 0.0, "formatted": True, "correct": correct}

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, texts))


def reward(text, task):
    return reward_batch([text], task)[0]


if __name__ == "__main__":
    tasks, n = build_tasks(pool())
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=1)
    by_split = {s: sum(t["split"] == s for t in tasks) for s in ("train", "eval")}
    print(f"{n} mutants generated, {len(tasks)} kept (bug visible in tests): {by_split}")
    t = random.Random(1).choice(tasks)
    print(f"\nexample {t['id']} ({t['mutation']}):\n{t['buggy']}\n--- visible test output ---\n{t['visible_output']}")
