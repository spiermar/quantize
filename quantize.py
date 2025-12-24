import argparse
import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Download a model in FP16 and quantize/load it in INT8 or AWQ.")
    parser.add_argument("model_id", type=str, help="The Hugging Face model ID (e.g., 'facebook/opt-350m').")
    parser.add_argument("--format", type=str, default="int8", choices=["int8", "awq", "w8a8"], help="Quantization format: 'int8' (bitsandbytes), 'awq' (AutoAWQ), or 'w8a8' (llmcompressor INT8).")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache downloaded model weights.")
    parser.add_argument("--save_dir", type=str, default="./quantized_model", help="Directory to save the quantized model.")
    parser.add_argument("--calib_dataset", type=str, default="open_platypus", help="Calibration dataset for W8A8 (default: open_platypus).")
    parser.add_argument("--calib_samples", type=int, default=512, help="Number of calibration samples for W8A8 (default: 512).")
    
    args = parser.parse_args()
    
    model_id = args.model_id
    
    # Check for 'AWS' vs 'AWQ' confusion
    if args.format.lower() == 'aws': 
        pass

    if args.format == 'w8a8':
        print(f"Converting '{model_id}' to W8A8 (INT8 weights + activations) using llmcompressor...")
        
        if not torch.cuda.is_available():
            print("Error: W8A8 quantization with llmcompressor typically requires a CUDA GPU.")
            return

        try:
            from llmcompressor.modifiers.quantization import GPTQModifier
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
            from llmcompressor.transformers import oneshot
            
            print(f"Using calibration dataset: {args.calib_dataset}")
            
            # Define the W8A8 recipe
            recipe = [
                SmoothQuantModifier(smoothing_strength=0.8),
                GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
            ]
            
            print("Starting one-shot quantization (this may take a significant amount of time)...")
            
            oneshot(
                model=model_id,
                dataset=args.calib_dataset,
                recipe=recipe,
                output_dir=args.save_dir,
                max_seq_length=2048,
                num_calibration_samples=args.calib_samples,
            )
            
            print(f"Success! W8A8 model saved to '{args.save_dir}'.")
            print("You can load this model in vLLM.")
            
        except ImportError:
            print("Error: 'llmcompressor' library not found. Please run: pip install llmcompressor")
        except Exception as e:
            print(f"An error occurred during W8A8 quantization: {e}")
            
        return

    if args.format == 'awq':
        print(f"Converting '{model_id}' to AWQ (W4A16) using llmcompressor...")
        print("Note: Assuming 'AWS' was a typo for 'AWQ'.")
        
        if not torch.cuda.is_available():
            print("Error: Quantization with llmcompressor typically requires a CUDA GPU.")
            return

        try:
            from llmcompressor.modifiers.quantization import GPTQModifier
            from llmcompressor.transformers import oneshot
            
            print(f"Using calibration dataset: {args.calib_dataset}")
            
            # Define the W4A16 (AWQ-like) recipe
            # Using GPTQ algorithm for 4-bit weight-only quantization
            recipe = [
                GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]),
            ]
            
            print("Starting one-shot quantization (this may take a significant amount of time)...")
            
            oneshot(
                model=model_id,
                dataset=args.calib_dataset,
                recipe=recipe,
                output_dir=args.save_dir,
                max_seq_length=2048,
                num_calibration_samples=args.calib_samples,
            )
            
            print(f"Success! AWQ (W4A16) model saved to '{args.save_dir}'.")
            print("You can load this model in vLLM.")
            
        except ImportError:
            print("Error: 'llmcompressor' library not found. Please run: pip install llmcompressor")
        except Exception as e:
            print(f"An error occurred during AWQ quantization: {e}")
            
        return

    # Default INT8 (bitsandbytes) behavior
    print(f"Downloading and loading '{model_id}' in FP16 with INT8 quantization...")
    
    try:
        # Check compatibility
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available.")
            print("Standard 'bitsandbytes' 8-bit quantization requires a CUDA GPU.")
            print("On macOS (Apple Silicon) or CPU, this script may fail or require specific bitsandbytes versions/compilation.")
            print("If you are running this on a generic CPU machine, consider using 'torch.quantization.quantize_dynamic' instead.")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir)
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
        
        print("\nSuccess! Model loaded.")
        print(f"Model memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
        print("Device map:", model.hf_device_map)
        
        # Example inference to verify
        print("\nVerifying model with a simple generation...")
        input_text = "Hello, my name is"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            
        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=20)
            
        print(f"Input: {input_text}")
        print(f"Output: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")
        
        # SAVE LOGIC
        print(f"\nSaving INT8 quantized model to '{args.save_dir}'...")
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print("Success! INT8 model saved.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nMake sure you have installed the requirements: pip install -r requirements.txt")
        print("Note: 'bitsandbytes' requires a CUDA-capable GPU.")

if __name__ == "__main__":
    main()
