# Coreference Resolution

## Model
The pre-trained coreference resolution model included here is the multilingual seq2seq transition-based model by [Bohnet et al. (2022)](https://arxiv.org/abs/2211.12142), accessed via [`google-research/coref_mt5`](https://github.com/google-research/google-research/blob/master/coref_mt5/README.md). 

### Model checkpoint download
The [model checkpoint linked to from `google-research/coref_mt5`](https://console.cloud.google.com/storage/browser/gresearch/correference_seq2seq/checkpoint_1140000) was downloaded with [`gsutil`](https://docs.cloud.google.com/storage/docs/gsutil_install), e.g.:
```
mkdir -p models/coref/coref_mt5_source

gsutil -m cp -r \
  gs://gresearch/correference_seq2seq/checkpoint_1140000/ \
  models/coref/coref_mt5_source/
```

The final [model config](./coref-mt5-xxl-bohnetetal2022/config.json) is saved in this project repository, but if re-downloading and re-converting the checkpoint from the source, the `mt5` model `config.json` must also be downloaded from HuggingFace:
```
curl -L \
  -o models/coref/coref-mt5-xxl-bohnetetal2022/mt5_xxl_config.json \
  https://huggingface.co/google/mt5-xxl/resolve/main/config.json
```

### Checkpoint Conversion
This checkpoint in T5X/JAX (Zarr) format was converted to a HuggingFace PyTorch checkpoint using [`convert_checkpoint.py`](./setup/convert_checkpoint.py).

First the virtual environment needs to be set up:
```
cd setup
./setup_checkpoint_conversion.sh
```

The resulting environment needs to be activated:
```
source .venv/bin/activate
```

Then the conversion script can be run:
```
cd ..

python setup/convert_checkpoint.py \
    --t5x_checkpoint_path coref_mt5_source/checkpoint_1140000/ \
    --config_file coref-mt5-xxl-bohnetetal2022/mt5_xxl_config.json \
    --pytorch_dump_path coref-mt5-xxl-bohnetetal2022
```

The above conversion results in two files in [`coref-mt5-xxl-bohnetetal2022`](./coref-mt5-xxl-bohnetetal2022) which can then be used to load the model with PyTorch:
* [`config.json`](./coref-mt5-xxl-bohnetetal2022/config.json)
* [`model.safetensors`](./coref-mt5-xxl-bohnetetal2022/model.safetensors)

## Usage
Coreference resolution can be run on novel texts using the [`coreference.py`](../../coreference.py) script in the project root directory, adapted from the [Jupyter notebook provided with the pre-trained model](https://github.com/google-research/google-research/blob/master/coref_mt5/colabs/coreference_decoder.ipynb).

The script should be run using the virtual environment from the main project root.

Example:
```
python3 coreference.py --model-path models/coref/coref-mt5-xxl-bohnetetal2022 --text "Der alten Entwicklungshilfepolitik der SPD hält er \"\"Wirtschaftsphobie, Bundeswehrphobie und Israelphobie \"\" vor. Israel, betont der neue Entwicklungshilfeminister, habe die weltweit besten Bewässerungssysteme. Die werde er in der arabischen Welt propagieren."
```

Example output with resolved coreference annotations:
```
=== document ===
Der alten Entwicklungshilfepolitik der SPD hält [1 er ] "" Wirtschaftsphobie , Bundeswehrphobie und Israelphobie "" vor . Israel , betont [1 der neue Entwicklungshilfeminister ] , habe [2 die weltweit besten Bewässerungssysteme ] . [2 Die ] werde [1 er ] in der arabischen Welt propagieren .
```
