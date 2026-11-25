# minirl

A minimal RL pipeline for coding tasks, written to learn how RL for LLMs
works end to end:

- **GRPO from scratch** on MLX, training LoRA adapters on
  `Qwen2.5-Coder-0.5B-Instruct`
- **SWE-style bug-fix tasks**: take correct code, reintroduce a bug, keep the
  task only if the tests catch it, and grade with hidden tests
- **A locked-down Docker sandbox** to run model-written code
- **Difficulty calibration**, so training only sees tasks that teach something

## GRPO in one loop

```
for each prompt: sample G answers                 # rollouts
score each answer                                 # e.g. hidden tests in the sandbox
advantage = (score - group mean) / group std
loss = -advantage * log P(tokens) + beta * KL(model || reference)
```

A group where every answer gets the same score has zero advantage, so it
teaches nothing. That's why tasks must be solved sometimes, not always or
never. `minirl/grpo.py` is commented as a walkthrough.

## Tasks

20 seed functions, each with visible and hidden tests. Every single-edit
mutation is tried (`<` to `<=`, `+` to `-`, `and` to `or`, constants off by
one). A mutant becomes a task only if the original passes and the visible
tests fail. 141 mutants gave 105 tasks; 29 of them come from 4 functions held
out for evaluation.

## What I learned

- **Calibration decides whether anything is learned.** The untrained model
  solved 5% of 2-digit x 2-digit products, so most groups gave no gradient.
  Adding easier problems raised that to 19%, with signal in 9 of 16 groups.
- **The model optimises what you measure.** Arithmetic accuracy went from
  52% to 60%, but answers shrank to a bare `ANSWER: 1234`, because the
  reward only checked the last line.
- **One noisy step can collapse the policy.** With the LoRA scale at 20, one
  update dropped accuracy from 50% to 12%. Scale 2 with 8 groups per step was
  stable.
- **Coding tasks:** the base model solved 37% of attempts. Of 76 training
  tasks, 61 were in the learnable band.

A full GRPO run on the coding tasks wasn't finished; the laptop was too slow.

## Running

Needs Apple Silicon (MLX) and Docker. The model (~1 GB) downloads on first
use.

```bash
uv venv && uv pip install mlx mlx-lm
python -m minirl.tasks.code               # build + validate tasks
python calibrate.py code train            # base-model success per task
python train.py --task arith --steps 60 --groups 8
```

The tasks also power [tinyvmm](https://github.com/namanarora00/tinyvmm)'s
VM-fork vs Docker sandbox benchmark.

## License

MIT, see [LICENSE](LICENSE).
