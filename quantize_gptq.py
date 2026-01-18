import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a Hugging Face causal LM with llmcompressor GPTQ."
    )
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., 'facebook/opt-350m').")
    parser.add_argument(
        "--format",
        choices=["w8a8", "w4a16"],
        default="w8a8",
        help="Target quantization: W8A8 for int8 or W4A16 for int4.",
    )
    parser.add_argument(
        "--save_dir",
        default="./quantized_model_gptq",
        help="Directory to save the quantized model.",
    )
    parser.add_argument(
        "--calib_dataset",
        default="open_platypus",
        help="Calibration dataset name/path for llmcompressor oneshot.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=512,
        help="Number of calibration samples used during oneshot quantization.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Max sequence length for quantization calibration.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA GPU is strongly recommended for llmcompressor GPTQ quantization.")

    try:
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor import oneshot
    except ImportError:
        print("Error: llmcompressor not installed. Run `pip install llmcompressor` and try again.")
        return

    recipe = []
    if args.format == "w8a8":
        recipe.append(SmoothQuantModifier(smoothing_strength=0.8))
        recipe.append(
            GPTQModifier(
                scheme="W8A8",
                targets="Linear",
                ignore=["lm_head"],
            )
        )
        print("Running W8A8 (int8) recipe.")
    else:
        recipe.append(
            GPTQModifier(
                scheme="W4A16",
                targets="Linear",
                ignore=["lm_head"],
            )
        )
        print("Running W4A16 (int4) recipe.")

    try:
        print(
            f"Starting oneshot quantization for '{args.model_id}' -> '{args.save_dir}', dataset '{args.calib_dataset}'."
        )
        oneshot(
            model=args.model_id,
            dataset=args.calib_dataset,
            recipe=recipe,
            output_dir=args.save_dir,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.calib_samples,
        )
        print(f"Quantized model saved to {args.save_dir}.")
        print("You can load the result with llmcompressor/AutoAWQ or vLLM.")
    except Exception as exc:
        print(f"Quantization failed: {exc}")
        print("Ensure CUDA is available and the calibration dataset is reachable.")


if __name__ == "__main__":
    main()
