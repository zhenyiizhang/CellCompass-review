# Getting started

Install CellCompass using the commands in the [README](../README.md), then run
all commands below from the repository directory with the `cellcompass`
environment active. The local checks used macOS. Linux is recommended for CUDA
training. Windows users should use WSL2.

## Configure the agent

Start the browser interface:

```bash
cellcompass --llm-model gpt-5.5 --llm-thinking-level high web --host 127.0.0.1 --port 8000 --device cpu
```

Open http://localhost:8000. The setup screen lets you select a provider and model
and choose a directory for results. For API access, select **OpenAI** and enter
an API key, or select an OpenAI-compatible provider and enter its API base URL.
For Codex access, follow the sign-in instructions below instead.

Use model **gpt-5.5** and, under **Advanced settings → Model auth & details**,
set **Reasoning effort** to **High**. The command above sets both options
explicitly. Other tool-calling models can be selected if needed.
Click **Initialize Agent** to open
the conversation. For algorithm design, also enable the stop hook described below.

CellCompass saves configuration and conversations under `~/.cellcompass/`.
Keep this directory private. The agent can execute analysis code and create files
on the machine where it runs. Use a dedicated study directory and keep the server
bound to localhost. The current interface is intended for one user at a time.

To train on an NVIDIA GPU, replace `--device cpu` with `--device cuda`.
Check that your PyTorch installation detects the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Codex sign-in

Run the following in an interactive terminal on the machine running CellCompass:

```bash
cellcompass auth codex-login
```

Open the URL printed in the terminal and complete sign-in. If prompted, paste
the returned code or redirect URL into the terminal. Node.js is required, and
the login helper installs its dependencies on first use. Check the saved login:

```bash
cellcompass auth profiles status
```

In the setup screen, select **OpenAI Codex OAuth**, model **gpt-5.5**, and
**Reasoning effort: High**. Under advanced settings, use **Auth mode: Codex OAuth**.
No API key is needed for this route. The account must have access to the selected
model. Account limits still apply. OpenAI distinguishes subscription sign-in from
API-key access in its [authentication documentation](https://learn.chatgpt.com/docs/auth).

The same settings can be passed when starting CellCompass:

```bash
cellcompass --llm-provider openai-codex --llm-auth-mode codex_oauth --llm-model gpt-5.5 --llm-thinking-level high web --host 127.0.0.1 --port 8000 --device cpu
```

## Algorithm-design settings

Before starting a model-design study, open **Advanced settings → Review and stop
hook**. Switch **Planner stop hook** on, choose **LLM prompt evaluator**, and
leave **Max triggers** at **20**. Keep the supplied stop-hook instructions, which
check whether the proposal, implementation, staged evaluations and final
confirmation have been completed.

When the agent tries to finish with work still outstanding, the hook directs it
to continue. It is not an unlimited loop: reaching the trigger limit pauses the
run for review. The **Stop** button can interrupt the run at any time.
Continuation also consumes model usage. Ordinary data inspection does not need
the hook.

These settings are saved with the model configuration. For terminal use, enabling
the hook explicitly is also possible after configuring its mode in the interface:

```bash
CYTOBRIDGE_STOP_HOOK_ENABLED=1 cellcompass --llm-model gpt-5.5 --llm-thinking-level high interactive /path/to/data.h5ad --output ./output/my_study --device cpu
```

## Try a small example

Create a synthetic time course before starting the server. If the server is
already running, open a second terminal, activate `cellcompass` and return to
the repository directory:

```bash
python examples/make_demo_data.py
```

The script prints the full path to `output/demo/toy_time_course.h5ad`.
It contains 600 simulated cells sampled at three times. The population separates
into two groups. These are artificial expression measurements for learning how
to use the software, not a biological study or a manuscript benchmark.

Paste the printed path into the conversation with this request:

> Load the synthetic time-course dataset at [paste the full path here]. Check
> the sampling times and plot the cells in the supplied latent coordinates.
> Describe the population changes and suggest an existing dynamical model.
> Explain how you would check its predictions before running any training.

This first step lets you check the data-loading and analysis workflow before a
longer fit. You should be able to inspect the three sampled populations and the
agent's proposed model and evaluation. You can then ask it to fit a small model
on CPU and compare generated populations with the observed data. Run time and
API usage depend on the model and the analysis requested.

To check the installation without making language-model calls:

```bash
python review/check_installation.py
```

## Use your own data

Provide a local AnnData `.h5ad` file and explain the experiment: the sampling
times, conditions, available measurements and the biological question. If the
CellCompass server runs on a remote machine, the file must be accessible there.

The standard training workflow uses `adata.obs['time_point_processed']` for
numeric sampling times and `adata.obsm['X_latent']` for the cell-state coordinates.
If your file has not been prepared this way, ask CellCompass to inspect it and
propose preprocessing first. RNA labeling, clone barcodes and paired protein
measurements require additional fields, described with the
[study implementations](../review/REVIEWER_GUIDE.md).

State which data should be used for training and which should be reserved for
evaluation. CellCompass can then assess whether an existing method is sufficient
or a new model is needed. Results are written to the output directory selected
in the setup screen, and saved conversations can be reopened from the interface.

## Terminal and remote use

After configuring a provider in the browser, you can also start an interactive
terminal session with `cellcompass`. Useful commands include:

```bash
cellcompass inspect /path/to/data.h5ad
cellcompass interactive /path/to/data.h5ad --output ./output/my_study --device cpu
cellcompass continue
cellcompass --help
```

For a server or cluster, run the web command there and forward its port from your
own computer:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-server
```

Then open http://localhost:8000 on your computer. Node.js is needed for the
browser build and Codex sign-in, but not for API-key-based terminal use. Optional literature
search requires a separately prepared local literature collection.

## If something does not start

A missing interface usually means the frontend has not been built. Run
`npm --prefix web ci` and `npm --prefix web run build`, then restart the server.
For a model connection error, check the provider, model name, API key and base URL
in the setup screen. For a CUDA error, first try `--device cpu` and check your
PyTorch and driver installation separately. API usage is billed by your provider.
