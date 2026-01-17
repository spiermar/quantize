import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face causal LM and quantize it with bitsandbytes."
    )
    parser.add_argument("model_id", help="Hugging Face model ID (e.g., 'gpt2').")
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Target bits for quantization (4 or 8).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the downloaded model weights.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./quantized_model",
        help="Directory to save the quantized model/tokenizer.",
    )
    args = parser.parse_args()

    quant_kwargs = {
        "device_map": "auto",
        "cache_dir": args.cache_dir,
        "trust_remote_code": True,
    }

    if args.bits == 8:
        quant_kwargs["load_in_8bit"] = True
        print(f"Loading '{args.model_id}' with 8-bit bitsandbytes quantization...")
    else:
        quant_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
        )
        print(f"Loading '{args.model_id}' with 4-bit bitsandbytes quantization (NF4)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            **quant_kwargs,
        )

        print("Model loaded.")
        if hasattr(model, "get_memory_footprint"):
            print(
                f"Memory footprint (GB): {model.get_memory_footprint() / 1024**3:.2f}"
            )

        example_prompt = "Hello, my name is"
        inputs = tokenizer(example_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model.to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20)

        print("Sample output:")
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        print(f"Saving quantized model/tokenizer to '{args.save_dir}'...")
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print("Saved quantized weights.")

    except Exception as exc:
        print(f"Failed to quantize/load the model: {exc}")
        print("Make sure bitsandbytes is installed and a CUDA-visible device is available.")


if __name__ == "__main__":
    main()
