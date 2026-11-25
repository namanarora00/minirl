# minirl

A minimal, readable RL pipeline for coding tasks, built to learn how RL
for LLMs works end to end:

- **GRPO written from scratch** on Apple's MLX, training LoRA adapters on
  `Qwen2.5-Coder-0.5B-Instruct`.
- **Bug-fixing environments**, built the way SWE-style RL environments are:
  start from correct code, reintroduce a bug, keep the task only if the
  tests catch it, and grade with hidden tests.
- **A hardened Docker sandbox** to run model-written code.
- **Difficulty calibration**, to keep only tasks that can produce a learning
  signal.

The tasks and test runner are also what
[tinyvmm](https://github.com/namanarora00/tinyvmm) uses to compare VM forks
against Docker as RL sandboxes.

## How a GRPO step works

```
repeat:
    take a batch of prompts
    for each prompt: sample G answers from the current model   <- rollouts, the slow part
    score each answer                                          <- e.g. run hidden tests in the sandbox
    advantage = (score - group mean) / group std               <- better than its siblings = positive
    loss = -advantage * log P(each generated token) + beta * KL(model || frozen reference)
    one optimizer step
```

A group where every answer scores the same has zero advantage everywhere,
and teaches nothing. That's why tasks are calibrated: the model should solve
them sometimes, not always or never. The code in `minirl/grpo.py` is
commented as a walkthrough.

## What's inside

| | |
|---|---|
| `minirl/model.py` | LoRA policy (rank 16), frozen reference, a batched group sampler (G copies of one prompt, so no padding), log-probs that project only completion positions to the 152k vocabulary |
| `minirl/grpo.py` | group advantages, per-token KL (k3 estimator, beta 0.04), one update per batch |
| `minirl/tasks/arith.py` | sanity task: 2-digit x 1- or 2-digit multiplication; the answer must be on a final `ANSWER: n` line |
| `minirl/tasks/code_seeds.py` | 20 small correct functions with visible and hidden tests |
| `minirl/tasks/code.py` | builds the bug-fixing tasks from the seeds, and grades answers in the sandbox |
| `minirl/sandbox/docker.py` | a pool of locked-down containers: no network, read-only root, tmpfs workdir, memory/CPU/process limits, timeouts |
| `calibrate.py` | samples G answers per task and keeps the tasks solved between 1 and G-1 times |
| `train.py` | the training loop, with greedy evaluation on a held-out set |

**How the coding tasks are built.** Each seed function is parsed, and every
single-edit mutation is generated: `<` to `<=`, `+` to `-`, `and` to `or`,
`in` to `not in`, a constant 0/1/2 off by one. A mutant becomes a task only
if the original code passes every test and the mutant fails at least one
*visible* test. The model sees the buggy file and the failing test output,
and replies with a fixed file. The reward is 1 if the *hidden* tests (visible
plus edge cases) all pass. 141 mutants gave 105 tasks: 76 for training, and
29 from four seed functions that are held out entirely, to test
generalisation.

## What I measured

- **Calibration decides whether anything is learned.** The untrained model
  solved 5% of 2-digit x 2-digit multiplications, so 7 of 10 groups were all
  wrong and produced no gradient. Mixing in 2-digit x 1-digit problems gave
  19% and signal in 9 of 16 groups.
- **The model optimises what you measure.** GRPO raised held-out greedy
  accuracy on arithmetic from 52% to 60% in 20 steps. It got there by
  dropping all the working: answers shrank to about 8 tokens
  (`ANSWER: 1234`), because the reward only checked the last line.
- **One noisy step can collapse the policy.** With the LoRA scale at 20, a
  single update driven by one lucky short answer dropped eval accuracy from
  50% to 12%. Scale 2 with 8 groups per step trained stably.
- **Coding tasks:** the untrained model solved 37% of attempts. Of 76
  training tasks, 14 were never solved, 61 were learnable, and 1 was always
  solved.

Not done: a full GRPO run on the coding tasks. It trains end to end
(`train.py --task code`), but a 16 GB laptop doing it alongside other work
was too slow to finish.

## Running it

Needs an Apple Silicon Mac (MLX) and Docker (Colima works) for the coding
tasks. The model weights (~1 GB) download from Hugging Face on first use.

```bash
uv venv && uv pip install mlx mlx-lm

python -m minirl.tasks.code                 # rebuild and validate the tasks in the sandbox
python calibrate.py code train              # base-model success per task -> runs/calibration-train.json
python train.py --task arith --steps 60 --groups 8

# train on the learnable coding tasks only
export MINIRL_TASK_IDS=$(python -c "import json; print(','.join(json.load(open('runs/calibration-train.json'))['learnable']))")
python train.py --task code --steps 40 --groups 6
```

Each run writes `runs/<name>/metrics.jsonl` (reward, accuracy, groups with
signal, completion length, KL, rollout vs training time per step) and the
trained LoRA adapters.

## Limitations

- Single-turn episodes: the model sees the failing tests once and answers
  with a full file. No multi-turn tool use (run tests, edit, retry) yet.
- The tasks are tiny functions. Real SWE environments are repos with
  dependencies, services and long setup.
- The Docker pool reuses containers across episodes, which is fast but
  weakens isolation. A fresh environment per episode is what
  [tinyvmm](https://github.com/namanarora00/tinyvmm) is about.
- MLX only, so Apple Silicon only.
