# CellCompass

CellCompass designs dynamical models for single-cell biology. Starting with a
biological question and experimental data, it selects an existing model or
develops a new method when the available models cannot represent what the
experiment measures. It carries the study through model fitting, evaluation and
biological analysis, and retains supported methods for use in later studies.

## Installation

You will need Conda and Node.js 22.12 or later. The commands below create a Python
3.10 environment. An API key for a supported language-model
provider is needed to use the agent. A CPU is enough to explore the interface and
small examples. We recommend an NVIDIA GPU for biological model training.

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
cellcompass web --host 127.0.0.1 --port 8000 --device cpu
```

Open [localhost:8000](http://localhost:8000). In the setup screen, choose your
model provider, enter the model name and API key, and click **Initialize Agent**.
Then describe your question and give CellCompass the path to your data. For example:

> My time-course scRNA-seq data are in /path/to/data.h5ad. How does this cell
> population change over time? Check the data and sampling times, then propose
> a dynamical model and explain how you would evaluate it before starting training.

For a first try without your own dataset, follow the
[small example](docs/getting-started.md#try-a-small-example).
The [getting-started guide](docs/getting-started.md) also covers input data,
GPU use and the terminal interface. Language-model calls use your provider account.

## Models and benchmarks

The repository includes the CellCompass agent, its dynamical engine, DynBench100
simulation and evaluation code, and the PulseBridge, LineageBridge, ProteinBridge
and BRGU-SFM implementations. See [models and data](review/REVIEWER_GUIDE.md)
for the relevant code and study datasets.

## Reproduce the figures

The [figure reproduction guide](figure_reproduction/README.md) redraws the six
main figures and fourteen supplementary figures from the included analysis data.
This runs separately from the agent and does not require an API key.
