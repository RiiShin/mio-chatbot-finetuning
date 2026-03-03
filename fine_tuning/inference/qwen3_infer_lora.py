#!/usr/bin/env python3
"""
Multi-round conversation inference script for Qwen3 model.
Processes conversations from JSON file and generates model responses.
"""

import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

def load_conversations(json_path):
    """Load conversations from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_conversation(conversations_data, tokenizer, model, device):
    """Process a single conversation and generate model responses."""
    system_prompt = conversations_data.get('system', '')
    conversations = conversations_data.get('conversations', [])
    
    # Initialize messages with system prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Process each turn
    output_conversations = []
    turn_count = 0
    
    for i, turn in enumerate(conversations):
        # if turn['role'] == 'user' or turn['from'] == 'human':
        if turn['from'] == 'human':
            turn_count += 1
            # Add human message
            messages.append({"role": "user", "content": turn['value']})
            output_conversations.append({
                "role": "user",
                "content": turn['value']
            })
            
            # Generate model response
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode
            )
            model_inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # Generate with specified parameters
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=32768,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0)
            
            # Extract only the new tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            model_response = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Add model response to output
            output_conversations.append({
                "role": "assistant",
                "content": model_response
            })
            
            # Add model response to messages for next round (use model's output, not ground truth)
            messages.append({"role": "assistant", "content": model_response})
            
        # elif turn['role'] == 'assistant' or turn['from'] == 'gpt':
        elif turn['from'] == 'gpt':
            # Skip ground truth, we already generated our own response
            pass
    
    # After all turns are finished, get the complete prompt text and its length
    # This includes system prompt + all human turns + all assistant responses
    complete_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    complete_input_ids = tokenizer([complete_text], return_tensors="pt").to(device)
    complete_length = complete_input_ids.input_ids.shape[1]
    
    print(f"\n{'='*60}")
    print(f"Complete input after {turn_count} turns:")
    print(f"Total tokens: {complete_length}")
    print(f"{'='*60}\n")
    
    return output_conversations

def main():
    parser = argparse.ArgumentParser(description='Multi-round conversation inference for Qwen3')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to input JSON file with conversations')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to output JSON file')
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

    
    print(f"Loading conversations from: {args.input_json}")
    conversations_data = load_conversations(args.input_json)
    
    print(f"Processing {len(conversations_data)} conversations...")
    output_data = []
    
    for idx, conv_data in enumerate(conversations_data):
        print(f"\n{'='*60}")
        print(f"Processing conversation {idx + 1}/{len(conversations_data)}")
        print(f"{'='*60}")
        
        output_conversations = process_conversation(conv_data, tokenizer, model, device)
        
        output_item = {
            "conversations": output_conversations,
            "tools": conv_data.get("tools", "[]"),
            "system": conv_data.get("system", "")
        }
        output_data.append(output_item)
    
    print(f"\nSaving results to: {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")

if __name__ == '__main__':
    main()
