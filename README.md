# quantize

This repo provides three helper scripts to download Hugging Face causal language models, quantize them with either `bitsandbytes` or `llmcompressor`, and persist the quantized artifacts.

## Scripts

- `quantize_bitsandbytes.py` – download a model, quantize it with bitsandbytes (4-bit NF4 or 8-bit), run a quick generation to sanity-check the weights, and save the tokenizer + quantized model to a directory. Key options:
  - `model_id` (positional): Hugging Face model identifier.
  - `--bits {4,8}` (default `8`): target quantization granularity.
  - `--save_dir`: destination for the quantized checkpoint.

- `quantize_gptq.py` – run llmcompressor’s GPTQ recipes to build W8A8 (int8) or W4A16 (int4) quantized weights via the `oneshot` pipeline. Supports calibration dataset, sample count, sequence length, and save directory, and warns if CUDA is missing.

- `quantize_awq.py` – use llmcompressor’s AWQ modifier (not GPTQ) to create AWQ-style weights in W8A8 (int8) or W4A16 (int4) format, again using the oneshot API with dataset/calibration controls.

## Requirements

- Python 3.10+ (per `.python-version`)
- `torch`, `transformers`, `bitsandbytes` (when running `quantize_bitsandbytes.py`)
- `llmcompressor` (for `quantize_gptq.py` and `quantize_awq.py`)

Install with:

```bash
pip install torch transformers bitsandbytes llmcompressor
```

## Examples

Save an 8-bit bitsandbytes quantized model:

```bash
python quantize_bitsandbytes.py facebook/opt-350m --bits 8 --save_dir ./run/bnb-8bit
```

Run GPTQ W4A16 quantization:

```bash
python quantize_gptq.py facebook/opt-350m --format w4a16 --save_dir ./run/gptq-w4a16
```

Run AWQ-style W8A8 quantization:

```bash
python quantize_awq.py facebook/opt-350m --format w8a8 --calib_dataset open_platypus
```

Each script prints progress and saves the quantized model/tokenizer (bitsandbytes) or compressed artifacts (llmcompressor) under `--save_dir`.
