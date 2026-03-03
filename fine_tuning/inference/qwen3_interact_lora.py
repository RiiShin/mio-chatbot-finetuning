#!/usr/bin/env python3
"""
Interactive multi-round conversation inference script for Qwen3 model with LoRA.
Reads system prompt from the first element of a JSON file, then enters interactive mode.
"""

import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser(description='Interactive multi-round conversation inference for Qwen3')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to input JSON file (reads system prompt from first element)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B',
                        help='Model name or path')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--lora_path', type=str, required=True, help='Path to LoRA checkpoint')

    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")

    # 加载基础模型和LoRA权重
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )


    print(f"Loading LoRA weights from: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    print(f"Model dtype: {model.dtype}")
    
    device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model loaded on device: {device}")

    # INSERT_YOUR_CODE
    # Print the model/tokenizer's max_input_len if available
    # For Qwen/Qwen3 models, this is usually tokenizer.model_max_length or tokenizer.max_input_len
    # Try both; print whichever is available
    max_input_len = getattr(tokenizer, "max_input_len", None)
    if max_input_len is None:
        max_input_len = getattr(tokenizer, "model_max_length", None)
    print(f"tokenizer.max_input_len/model_max_length: {max_input_len}")

    # Load system prompt from first element of JSON
    print(f"Loading system prompt from: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    system_prompt = data[0].get('system', '')

    # Initialize messages with system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print(f"System prompt loaded ({len(system_prompt)} chars)")
    else:
        print("No system prompt found in first element")

    # Interactive loop
    print(f"\n{'='*60}")
    print("Entering interactive mode. Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to reset conversation (keep system prompt).")
    print(f"{'='*60}\n")

    turn_count = 0
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit'):
            print("Bye!")
            break
        if user_input.lower() == 'clear':
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            turn_count = 0
            print("[Conversation cleared]\n")
            continue

        turn_count += 1
        messages.append({"role": "user", "content": user_input})

        # Generate model response
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        model_response = tokenizer.decode(output_ids, skip_special_tokens=True)

        messages.append({"role": "assistant", "content": model_response})

        print(f"\nAssistant: {model_response}")
        print(f"  [turn {turn_count} | total tokens: {len(generated_ids[0])}]\n")

if __name__ == '__main__':
    main()
