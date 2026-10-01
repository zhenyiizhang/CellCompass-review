# Getting started

Install CellCompass using the commands in the [README](../README.md), then run
all commands below from the repository directory with the `cellcompass`
environment active. The local checks used macOS. Linux is recommended for CUDA
training. Windows users should use WSL2.

## Configure the agent

Start the browser interface:

```bash
cellcompass web --host 127.0.0.1 --port 8000 --device cpu
```

Open http://localhost:8000. The setup screen lets you select a provider and model,
enter an API key, and choose a directory for your results. For an OpenAI-compatible
provider, also enter its API base URL. Use a model that supports tool calling.
For example, select **OpenAI**, enter your API key, and use a tool-calling model
available to your account. Keep the provider's default base URL.
Click **Initialize Agent** to open the conversation.

CellCompass saves configuration and conversations under `~/.cellcompass/`.
Keep this directory private. The agent can execute analysis code and create files
on the machine where it runs. Use a dedicated study directory and keep the server
bound to localhost. The current interface is intended for one user at a time.

To train on an NVIDIA GPU, replace `--device cpu` with `--device cuda`.
Check that your PyTorch installation detects the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
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

Then open http://localhost:8000 on your computer. Node.js is needed to build the
browser interface, but not for API-key-based terminal use. Optional literature
search requires a separately prepared local literature collection.

## If something does not start

A missing interface usually means the frontend has not been built. Run
`npm --prefix web ci` and `npm --prefix web run build`, then restart the server.
For a model connection error, check the provider, model name, API key and base URL
in the setup screen. For a CUDA error, first try `--device cpu` and check your
PyTorch and driver installation separately. API usage is billed by your provider.
