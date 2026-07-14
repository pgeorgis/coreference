#!/usr/bin/env python3

"""Convert a T5X/JAX (Zarr) checkpoint to a HuggingFace PyTorch checkpoint.

Drop-in replacement for the HuggingFace convert_t5x_checkpoint_to_pytorch.py
that avoids the t5x and tensorflow dependencies and is memory-efficient enough
to run on a 64 GB machine for mT5-XXL.

Memory strategy
---------------
T5X stores parameters in two places:
  - Large arrays (attention, FFN weights): one Zarr directory per parameter.
  - Small arrays (layer norms, position biases): inline in the `checkpoint`
    MsgPack file as ExtType(code=1) objects (payload = msgpack [shape, dtype,
    raw_bytes]).

This script loads and converts ONE Zarr array at a time to float16, freeing
the float32 source immediately. Peak memory ≈ 24 GB (float16 output dict) +
the single largest tensor (~4 GB for the token embedding). Total ≈ 28 GB,
well within 64 GB.
"""

import argparse
import json
import os

import msgpack
import numpy as np
import torch
import zarr
from safetensors.torch import save_file as safetensors_save_file
from transformers import MT5Config

# ---------------------------------------------------------------------------
# Static parameter name mapping
# ---------------------------------------------------------------------------

def build_param_mapping(
    num_layers: int,
    num_decoder_layers: int,
    split_mlp_wi: bool,
    is_encoder_only: bool,
) -> dict[str, tuple[str, bool]]:
    """Return a dict mapping every T5X parameter path to (hf_name, needs_transpose).

    T5X stores weight matrices as (input_dim, output_dim); HuggingFace/PyTorch
    expects (output_dim, input_dim), so kernel matrices must be transposed.
    Layer norm scales and position-bias embeddings are not transposed.
    """
    m: dict[str, tuple[str, bool]] = {}

    # Shared token embeddings (Zarr, no transpose).
    m["token_embedder/embedding"] = ("shared.weight", False)

    # --- Encoder ---
    for i in range(num_layers):
        t = f"encoder/layers_{i}"   # t5x prefix
        h = f"encoder.block.{i}"   # hf prefix

        # Layer norms (inline in MsgPack, no transpose).
        m[f"{t}/pre_attention_layer_norm/scale"] = (f"{h}.layer.0.layer_norm.weight", False)
        m[f"{t}/pre_mlp_layer_norm/scale"]       = (f"{h}.layer.1.layer_norm.weight", False)

        # Self-attention (Zarr, transpose).
        for t5, hf in [("key", "k"), ("out", "o"), ("query", "q"), ("value", "v")]:
            m[f"{t}/attention/{t5}/kernel"] = (f"{h}.layer.0.SelfAttention.{hf}.weight", True)

        # FFN (Zarr, transpose).
        if split_mlp_wi:
            m[f"{t}/mlp/wi_0/kernel"] = (f"{h}.layer.1.DenseReluDense.wi_0.weight", True)
            m[f"{t}/mlp/wi_1/kernel"] = (f"{h}.layer.1.DenseReluDense.wi_1.weight", True)
        else:
            m[f"{t}/mlp/wi/kernel"]   = (f"{h}.layer.1.DenseReluDense.wi.weight", True)
        m[f"{t}/mlp/wo/kernel"]       = (f"{h}.layer.1.DenseReluDense.wo.weight", True)

    # Encoder-level (inline in MsgPack).
    m["encoder/encoder_norm/scale"]          = ("encoder.final_layer_norm.weight", False)
    m["encoder/relpos_bias/rel_embedding"]   = (
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", True
    )

    if is_encoder_only:
        return m

    # --- Decoder ---
    for i in range(num_decoder_layers):
        t = f"decoder/layers_{i}"
        h = f"decoder.block.{i}"

        # Layer norms (inline, no transpose).
        m[f"{t}/pre_self_attention_layer_norm/scale"]  = (f"{h}.layer.0.layer_norm.weight", False)
        m[f"{t}/pre_cross_attention_layer_norm/scale"] = (f"{h}.layer.1.layer_norm.weight", False)
        m[f"{t}/pre_mlp_layer_norm/scale"]             = (f"{h}.layer.2.layer_norm.weight", False)

        # Self-attention (Zarr, transpose).
        for t5, hf in [("key", "k"), ("out", "o"), ("query", "q"), ("value", "v")]:
            m[f"{t}/self_attention/{t5}/kernel"] = (f"{h}.layer.0.SelfAttention.{hf}.weight", True)

        # Cross-attention (Zarr, transpose).
        for t5, hf in [("key", "k"), ("out", "o"), ("query", "q"), ("value", "v")]:
            m[f"{t}/encoder_decoder_attention/{t5}/kernel"] = (
                f"{h}.layer.1.EncDecAttention.{hf}.weight", True
            )

        # FFN (Zarr, transpose).
        if split_mlp_wi:
            m[f"{t}/mlp/wi_0/kernel"] = (f"{h}.layer.2.DenseReluDense.wi_0.weight", True)
            m[f"{t}/mlp/wi_1/kernel"] = (f"{h}.layer.2.DenseReluDense.wi_1.weight", True)
        else:
            m[f"{t}/mlp/wi/kernel"]   = (f"{h}.layer.2.DenseReluDense.wi.weight", True)
        m[f"{t}/mlp/wo/kernel"]       = (f"{h}.layer.2.DenseReluDense.wo.weight", True)

    # Decoder-level (inline in MsgPack).
    m["decoder/decoder_norm/scale"]        = ("decoder.final_layer_norm.weight", False)
    m["decoder/relpos_bias/rel_embedding"] = (
        "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", True
    )
    # mT5 v1.1 has a separate lm_head (not tied to embeddings).
    m["decoder/logits_dense/kernel"]       = ("lm_head.weight", True)

    return m


# ---------------------------------------------------------------------------
# Checkpoint reading
# ---------------------------------------------------------------------------

def _decode_ext_array(ext: "msgpack.ext.ExtType") -> np.ndarray:
    """Decode a T5X inline array from a MsgPack ExtType(code=1) object.

    The payload is itself MsgPack-encoded as [shape, dtype_str, raw_bytes].
    """
    shape, dtype_str, data_bytes = msgpack.unpackb(ext.data, raw=False)
    return np.frombuffer(data_bytes, dtype=dtype_str).reshape(shape)


def load_inline_params(checkpoint_dir: str) -> dict[str, np.ndarray]:
    """Load small arrays stored inline in the MsgPack `checkpoint` file.

    These are layer norm scales, position biases, and similar small tensors
    that T5X embeds directly in the checkpoint metadata rather than as
    separate Zarr directories.

    Returns a flat dict keyed by slash-separated T5X parameter paths, e.g.:
        "decoder/decoder_norm/scale"
        "encoder/layers_0/pre_attention_layer_norm/scale"
    """
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MsgPack checkpoint file not found: {ckpt_path!r}")

    with open(ckpt_path, "rb") as f:
        ckpt = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

    params: dict[str, np.ndarray] = {}

    def _extract(obj: object, path: str = "") -> None:
        if isinstance(obj, msgpack.ext.ExtType):
            params[path] = _decode_ext_array(obj)
        elif isinstance(obj, dict):
            if obj.get("driver") == "zarr":
                return  # Zarr weights are loaded separately
            for k, v in obj.items():
                _extract(v, f"{path}/{k}" if path else k)

    _extract(ckpt["optimizer"]["target"])
    return params


def detect_split_mlp(checkpoint_dir: str) -> bool:
    """Return True if the checkpoint uses gated GeLU (wi_0 + wi_1) FFN layers."""
    return any(
        name == "target.encoder.layers_0.mlp.wi_0.kernel"
        for name in os.listdir(checkpoint_dir)
    )


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------

def stream_convert(
    checkpoint_dir: str,
    config: MT5Config,
    is_encoder_only: bool,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Convert T5X parameters to a HuggingFace state dict, one tensor at a time.

    Zarr arrays are loaded, transposed if needed, cast to `dtype`, and stored
    in the output dict. The float32 source array is deleted immediately after
    conversion, keeping peak memory low.

    Returns a complete state dict ready for safetensors.save_file().
    """
    split_mlp_wi = detect_split_mlp(checkpoint_dir)
    print(f"Gated GeLU FFN (split wi_0/wi_1): {split_mlp_wi}")

    mapping = build_param_mapping(
        config.num_layers,
        config.num_decoder_layers,
        split_mlp_wi,
        is_encoder_only,
    )

    state_dict: dict[str, torch.Tensor] = {}

    def _to_tensor(arr: np.ndarray, needs_T: bool) -> torch.Tensor:
        np_arr = arr.T if needs_T else arr
        return torch.from_numpy(np_arr.copy()).to(dtype)

    # --- 1. Inline arrays (layer norms, position biases) ---
    # These are small (~50 MB total) so loading them all at once is fine.
    print("Loading inline arrays (layer norms, position biases) ...")
    inline = load_inline_params(checkpoint_dir)
    n_inline = 0
    for t5x_key, arr in inline.items():
        if t5x_key in mapping:
            hf_name, needs_T = mapping[t5x_key]
            state_dict[hf_name] = _to_tensor(arr, needs_T)
            n_inline += 1
        # else: optimizer-state key or similar — ignore
    print(f"  {n_inline} inline arrays loaded.")

    # --- 2. Zarr arrays (large weight matrices) — streamed one at a time ---
    zarr_entries = sorted(
        name for name in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, name))
        and os.path.exists(os.path.join(checkpoint_dir, name, ".zarray"))
    )
    if not zarr_entries:
        raise FileNotFoundError(
            f"No Zarr parameter directories found in {checkpoint_dir!r}. "
            "Make sure you are passing the inner checkpoint directory that "
            "contains `target.*` sub-directories."
        )

    n_zarr = len(zarr_entries)
    print(f"Streaming {n_zarr} Zarr weight arrays (one at a time) ...")
    n_loaded = 0
    n_skipped = 0

    for idx, name in enumerate(zarr_entries):
        t5x_key = name.removeprefix("target.").replace(".", "/")

        if t5x_key not in mapping:
            n_skipped += 1
            continue

        hf_name, needs_T = mapping[t5x_key]

        # Load → optionally transpose → cast to target dtype → store.
        arr = zarr.open(os.path.join(checkpoint_dir, name), mode="r")[:]  # float32
        state_dict[hf_name] = _to_tensor(arr, needs_T)
        del arr  # free float32 immediately

        n_loaded += 1
        if n_loaded % 50 == 0 or idx + 1 == n_zarr:
            mb = sum(t.nbytes for t in state_dict.values()) / 1e6
            print(f"  [{idx + 1}/{n_zarr}] {n_loaded} loaded — "
                  f"state dict: {mb / 1e3:.1f} GB")

    if n_skipped:
        print(f"  {n_skipped} Zarr arrays had no HF mapping and were skipped.")

    # --- 3. Tied weight aliases ---
    # encoder.embed_tokens and decoder.embed_tokens both mirror shared.weight.
    # We clone here so safetensors.save_file() sees unique data pointers.
    # HuggingFace's tie_weights() will re-tie them on load, but the clones
    # ensure the safetensors file passes validation.
    emb = state_dict["shared.weight"]
    state_dict["encoder.embed_tokens.weight"] = emb.clone()
    if not is_encoder_only:
        state_dict["decoder.embed_tokens.weight"] = emb.clone()
        if "lm_head.weight" not in state_dict:
            print("No separate lm_head found — tying to shared embeddings.")
            state_dict["lm_head.weight"] = emb.clone()

    return state_dict


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_hf_checkpoint(
    state_dict: dict[str, torch.Tensor],
    config: MT5Config,
    pytorch_dump_path: str,
    dtype: torch.dtype,
) -> None:
    """Save the converted state dict and config in HuggingFace format."""
    os.makedirs(pytorch_dump_path, exist_ok=True)

    # Save weights.
    weights_path = os.path.join(pytorch_dump_path, "model.safetensors")
    print(f"Writing {weights_path} ...")
    safetensors_save_file(state_dict, weights_path)
    print("  Done.")

    # Save config. Annotate torch_dtype so from_pretrained(..., torch_dtype='auto')
    # loads in float16 without an explicit flag.
    dtype_str = {torch.float16: "float16", torch.float32: "float32",
                 torch.bfloat16: "bfloat16"}.get(dtype, str(dtype))
    config_dict = json.loads(config.to_json_string())
    config_dict["torch_dtype"] = dtype_str
    config_path = os.path.join(pytorch_dump_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Config saved to {config_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert(
    t5x_checkpoint_path: str,
    config_file: str,
    pytorch_dump_path: str,
    is_encoder_only: bool = False,
    dtype: torch.dtype = torch.float16,
) -> None:
    config = MT5Config.from_json_file(config_file)
    print(
        f"Config: {config.num_layers} encoder layers, "
        f"{config.num_decoder_layers} decoder layers, "
        f"d_model={config.d_model}, vocab={config.vocab_size}"
    )
    print(f"Target dtype: {dtype}")

    state_dict = stream_convert(t5x_checkpoint_path, config, is_encoder_only, dtype)

    total_params = sum(t.numel() for t in state_dict.values())
    total_gb = sum(t.nbytes for t in state_dict.values()) / 1e9
    print(f"\nState dict: {len(state_dict)} tensors, "
          f"~{total_params / 1e9:.1f}B parameters, {total_gb:.1f} GB")

    save_hf_checkpoint(state_dict, config, pytorch_dump_path, dtype)

    print(f"\nConversion complete. Load with:")
    print(f"  from transformers import MT5ForConditionalGeneration")
    print(f"  model = MT5ForConditionalGeneration.from_pretrained(")
    print(f"      '{pytorch_dump_path}', torch_dtype='auto')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert a T5X/JAX (Zarr) checkpoint to HuggingFace safetensors format. "
            "Streams one tensor at a time — peak memory ≈ 28 GB for mT5-XXL in float16. "
            "Requires: pip install zarr msgpack safetensors"
        )
    )
    parser.add_argument(
        "--t5x_checkpoint_path", required=True,
        help=(
            "Path to the INNER checkpoint directory containing `target.*` Zarr "
            "sub-directories and the `checkpoint` MsgPack file. "
            "For gsutil-downloaded checkpoints this is typically "
            "coref_mt5/models/checkpoint_1140000/"
        ),
    )
    parser.add_argument(
        "--config_file", required=True,
        help=(
            "Path to the HuggingFace model config.json. "
            "Download with: curl -L -o mt5_xxl_config.json "
            "https://huggingface.co/google/mt5-xxl/resolve/main/config.json"
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", required=True,
        help="Output directory for the converted HuggingFace checkpoint.",
    )
    parser.add_argument(
        "--is_encoder_only", action="store_true", default=False,
        help="Set if converting an encoder-only model.",
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
        help="Output weight dtype (default: float16, ~24 GB for mT5-XXL).",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    convert(
        args.t5x_checkpoint_path,
        args.config_file,
        args.pytorch_dump_path,
        args.is_encoder_only,
        dtype_map[args.dtype],
    )
