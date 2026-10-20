# Coreference Resolution

Coreference resolution with the multilingual seq2seq transition-based model by [Bohnet et al. (2022)](https://arxiv.org/abs/2211.12142) (mT5-XXL), accessed via [`google-research/coref_mt5`](https://github.com/google-research/google-research/blob/master/coref_mt5/README.md). Originally developed for a [German corpus study of pronouns](https://github.com/SFB1252/c07-dpron-korpus) within the [CRC 1252](https://sfb1252.uni-koeln.de/en/) at the University of Cologne (Universität zu Köln), Germany.

The library provides:
* inference with the converted model (`coref.run`), adapted from the [Jupyter notebook provided with the pre-trained model](https://github.com/google-research/google-research/blob/master/coref_mt5/colabs/coreference_decoder.ipynb), with batching, parallel preprocessing, and dynamic output-length control;
* checkpointing helpers (`coref.checkpoints`) for resuming long runs after a crash;
* [tools](./tools/) to convert the released T5X/JAX checkpoint to a HuggingFace PyTorch checkpoint.

```python
from coref import load_coreference_model, run

model = load_coreference_model("/path/to/coref-mt5-xxl-bohnetetal2022")  # or set $COREF_MODEL_DIR
result = run(documents={"doc1": "Anna sieht ihre Freundin.\nDie grüßt Anna."}, model=model)
print(result["doc1"]["text"])  # text with [N ...] cluster brackets
```

## Installation
This library requires Python 3.11 or later and is best managed with [uv](https://docs.astral.sh/uv/).

As a development environment:
```bash
uv sync
```

## Hugging Face access token
The mT5 tokenizer (`google/mt5-xxl`) tokenizer is fetched from the HuggingFace Hub, which works without an account. However, if you have created a user access token (see [here](https://huggingface.co/docs/hub/en/security-tokens)), it can optionally be used to authenticate requests, to silence warnings and improve rate limits.

For the command-line interface, create a file called `.env` in the directory you run it from (a template is in [`.env.example`](./.env.example); `.env` is git-ignored):
```
HF_ACCESS_TOKEN=hf_abcd
```
Replace `hf_abcd` with your actual token.

## Model
**No pretrained model is included in this repository** (the converted weights are ~30 GB). Obtain the model by [downloading and converting the checkpoint](#model-checkpoint-download-and-conversion), then either pass its directory to `load_coreference_model(model_path)`, or set the environment variable `COREF_MODEL_DIR`:
```bash
export COREF_MODEL_DIR=/path/to/pretrained/coref-mt5-xxl-bohnetetal2022
```

### Model checkpoint download and conversion
The [model checkpoint linked to from `google-research/coref_mt5`](https://console.cloud.google.com/storage/browser/gresearch/correference_seq2seq/checkpoint_1140000) can be downloaded with [`gsutil`](https://docs.cloud.google.com/storage/docs/gsutil_install) (~79 GB), from the repository root:
```bash
mkdir -p pretrained/coref_mt5_source

gsutil -m cp -r \
  gs://gresearch/correference_seq2seq/checkpoint_1140000/ \
  pretrained/coref_mt5_source/
```

If re-converting the checkpoint from the source, the `mt5` model `config.json` must also be downloaded from HuggingFace:
```bash
mkdir -p pretrained/coref-mt5-xxl-bohnetetal2022

curl -L \
  -o pretrained/coref-mt5-xxl-bohnetetal2022/mt5_xxl_config.json \
  https://huggingface.co/google/mt5-xxl/resolve/main/config.json
```

The checkpoint in T5X/JAX (Zarr) format is converted to a HuggingFace PyTorch checkpoint using [`convert_checkpoint.py`](./tools/convert_checkpoint.py). First set up the virtual environment, a separate environment used only for this one-time conversion (not the library's environment):
```bash
cd tools
./setup_checkpoint_conversion.sh
source .venv/bin/activate
cd ..
```
Then run the conversion script from the repository root:
```bash
python tools/convert_checkpoint.py \
    --t5x_checkpoint_path pretrained/coref_mt5_source/checkpoint_1140000/ \
    --config_file pretrained/coref-mt5-xxl-bohnetetal2022/mt5_xxl_config.json \
    --pytorch_dump_path pretrained/coref-mt5-xxl-bohnetetal2022
```
The conversion results in two files in `pretrained/coref-mt5-xxl-bohnetetal2022` which can then be used to load the model with PyTorch:
* `config.json`
* `model.safetensors`

The source checkpoint is only needed to repeat the conversion.

## Usage
Coreference resolution can be run on novel texts from the command line:
```bash
python -m coref --model-path pretrained/coref-mt5-xxl-bohnetetal2022 --text "Der alten Entwicklungshilfepolitik der SPD hält er \"\"Wirtschaftsphobie, Bundeswehrphobie und Israelphobie \"\" vor. Israel, betont der neue Entwicklungshilfeminister, habe die weltweit besten Bewässerungssysteme. Die werde er in der arabischen Welt propagieren."
```

Example output with resolved coreference annotations:
```
=== document ===
Der alten Entwicklungshilfepolitik der SPD hält [1 er ] "" Wirtschaftsphobie , Bundeswehrphobie und Israelphobie "" vor . Israel , betont [1 der neue Entwicklungshilfeminister ] , habe [2 die weltweit besten Bewässerungssysteme ] . [2 Die ] werde [1 er ] in der arabischen Welt propagieren .
```
Use `--file` for texts in files (one sentence per line), and `--pretokenized` for already tokenized input.

In Python, `coref.run(documents, model, batch_size=16, max_workers=1, is_tokenized=False, ...)` returns, for each document, the annotated `"text"` and the entity `"spans"`.

### Resuming long runs
`run` accepts a `checkpoint_callback(partial_results, milestone, in_progress_states)` that is called at each progress milestone, and `in_progress_states` to resume partially processed documents. The helpers in [`coref.checkpoints`](./src/coref/checkpoints.py) save and load such checkpoints next to an input file (`<input>.coref_checkpoint_<N>.json`), serialize and restore the in-progress `State`s, and clean up after a successful run.

## Repository structure
* [`src/coref/`](./src/coref/)
    * [`coreference.py`](./src/coref/coreference.py): model loading, the transition-based `State`, `run`, and the command-line interface.
    * [`checkpoints.py`](./src/coref/checkpoints.py): checkpoint helpers for resumable runs.
    * [`paths.py`](./src/coref/paths.py), [`constants.py`](./src/coref/constants.py), [`progress.py`](./src/coref/progress.py): model lookup, constants, and progress logging.
* [`tools/`](./tools/): checkpoint conversion script and environment setup.
* [`configs/`](./configs/): the converted model's `config.json`.
* [`tests/`](./tests/)

## License
Apache License 2.0, see [`LICENSE`](./LICENSE). Parts of the code are adapted from Google Research's [`coref_mt5`](https://github.com/google-research/google-research/tree/master/coref_mt5), see [`NOTICE`](./NOTICE). The model checkpoint is not part of this repository and is subject to the terms under which Google Research distributes it.
