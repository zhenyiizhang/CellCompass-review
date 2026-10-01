# CellCompass

CellCompass designs dynamical models for single-cell biology. Starting with a
biological question and experimental data, it selects an existing model or
develops a new method when the available models cannot represent what the
experiment measures. It carries the study through model fitting, evaluation and
biological analysis, and retains supported methods for use in later studies.

## Installation

Requirements: Conda, Node.js 22.12 or later, and either a supported provider's
API key or Codex sign-in. The installation uses Python 3.10. CPU execution is
supported for small examples. An NVIDIA GPU is recommended for biological model
training.

```bash
git clone https://github.com/zhenyiizhang/CellCompass-review.git
cd CellCompass-review
conda create -n cellcompass python=3.10 -y
conda activate cellcompass
python -m pip install -c requirements-runtime-constraints.txt -r requirements-agent.txt -e ./CytoBridge-main -e .
npm --prefix web ci
npm --prefix web run build
```

## Start CellCompass

```bash
cellcompass --llm-model gpt-5.5 --llm-thinking-level high web --host 127.0.0.1 --port 8000 --device cpu
```

The examples use GPT-5.5 with high reasoning effort. Open
[localhost:8000](http://localhost:8000) and configure the provider in the setup
screen. For API access, enter an API key. For Codex access, first follow the
[Codex sign-in instructions](docs/getting-started.md#codex-sign-in).
Click **Initialize Agent**, then describe your question and provide the data path:

> My time-course scRNA-seq data are in /path/to/data.h5ad. How does this cell
> population change over time? Check the data and sampling times, then propose
> a dynamical model and explain how you would evaluate it before starting training.

For a first try without your own dataset, follow the
[small example](docs/getting-started.md#try-a-small-example).
The [getting-started guide](docs/getting-started.md) also covers input data,
GPU use and the terminal interface. Language-model calls use your provider account.

## Algorithm design

For a model-design study, open **Advanced settings → Review and stop hook** and
enable **Planner stop hook** before initializing the agent. Select **LLM prompt
evaluator** and keep the supplied instructions. This checks unfinished work when
the agent tries to stop and directs it to continue model development and evaluation.
The hook is off in a fresh installation. See
[algorithm-design settings](docs/getting-started.md#algorithm-design-settings)
for the continuation limit and how to stop a run.

## Models and benchmarks

The repository includes the CellCompass agent, its dynamical engine, DynBench100
simulation and evaluation code, and the PulseBridge, LineageBridge, ProteinBridge
and BRGU-SFM implementations. See [models and data](review/REVIEWER_GUIDE.md)
for the relevant code and study datasets.

## Reproduce the figures

The [figure reproduction guide](figure_reproduction/README.md) redraws the six
main figures and fourteen supplementary figures from the included analysis data.
This runs separately from the agent and does not require an API key.
