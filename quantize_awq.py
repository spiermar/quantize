import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a Hugging Face causal LM via llmcompressor, emulating AWQ flows."
    )
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., 'gpt2').")
    parser.add_argument(
        "--format",
        choices=["w8a8", "w4a16"],
        default="w8a8",
        help="Quantize to W8A8 (int8) or W4A16 (int4) format.",
    )
    parser.add_argument(
        "--save_dir",
        default="./quantized_model_awq",
        help="Directory to persist the quantized artifacts.",
    )
    parser.add_argument(
        "--calib_dataset",
        default="open_platypus",
        help="Dataset identifier for llmcompressor oneshot calibration.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=512,
        help="Number of calibration samples for oneshot quantization.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length during calibration.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA GPUs improve llmcompressor AWQ results.")

    try:
        from llmcompressor.modifiers.quantization import AWQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.transformers import oneshot
    except ImportError:
        print("Error: llmcompressor is not installed. Run `pip install llmcompressor` and retry.")
        return

    recipe = []
    if args.format == "w8a8":
        print("Running W8A8 (INT8) AWQ recipe with SmoothQuant.")
        recipe.append(SmoothQuantModifier(smoothing_strength=0.8))
        recipe.append(
            AWQModifier(
                scheme="W8A8",
                targets="Linear",
                ignore=["lm_head"],
            )
        )
    else:
        print("Running W4A16 (INT4) AWQ recipe.")
        recipe.append(
            AWQModifier(
                scheme="W4A16",
                targets="Linear",
                ignore=["lm_head"],
            )
        )

    print(
        f"Starting AWQ quantization of '{args.model_id}' ({args.format}) and saving to '{args.save_dir}'."
    )

    try:
        oneshot(
            model=args.model_id,
            dataset=args.calib_dataset,
            recipe=recipe,
            output_dir=args.save_dir,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.calib_samples,
        )
        print(f"Quantized model stored at '{args.save_dir}'.")
        print("Use llmcompressor, AutoAWQ, or vLLM to load the result.")
    except Exception as exc:
        print(f"Quantization failed: {exc}")
        print("Check CUDA availability, dataset accessibility, and llmcompressor compatibility.")


if __name__ == "__main__":
    main()
