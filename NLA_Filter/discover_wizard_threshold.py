import os
import json
import yaml
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
AR_MODEL_PATH = "./models/nla-qwen-ar"  
INPUT_FILE = "wizard_500_sample.jsonl"
OUTPUT_FILE = "wizard_500_discovery.jsonl"

def load_nla_metadata(ar_path):
    meta_path = os.path.join(ar_path, "nla_meta.yaml")
    with open(meta_path, 'r') as f: return yaml.safe_load(f)

def main():
    print("Loading Base Tokenizer and AR Value Critic...")
    meta = load_nla_metadata(AR_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    critic = AutoModelForCausalLM.from_pretrained(
        AR_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
    ).eval()

    head_path = os.path.join(AR_MODEL_PATH, "value_head.safetensors")
    value_head_weights = load_file(head_path)
    v_weight = value_head_weights[list(value_head_weights.keys())[0]].to("cuda").to(torch.bfloat16)

    examples = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f: examples.append(json.loads(line))

    # PASS 1: Calculate Raw Scores
    with torch.no_grad():
        for ex in tqdm(examples, desc="Calculating Raw Scores"):
            prompt_templates = meta.get("prompt_templates", {})
            ar_template = prompt_templates.get("ar", "Summary of the following text: <text>{explanation}</text> <summary")
            formatted_text = ar_template.replace("{explanation}", ex["text"])
            
            inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
            outputs = critic(**inputs, output_hidden_states=True)
            
            final_token_state = outputs.hidden_states[-1][0, -1, :]
            projected_vector = torch.matmul(final_token_state, v_weight.T)
            raw_score = torch.norm(projected_vector).item()
            
            ex['raw_nla_score'] = raw_score * meta.get("mse_scale", 1.0)

    # PASS 2: Normalize and Save ALL
    raw_scores = [ex['raw_nla_score'] for ex in examples]
    min_score, max_score = min(raw_scores), max(raw_scores)
    
    print(f"\nDiscovered Raw Range: [{min_score:.4f}, {max_score:.4f}]")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for ex in examples:
            # Normalize to 0.0 - 1.0
            normalized = (ex['raw_nla_score'] - min_score) / (max_score - min_score)
            ex['nla_score_normalized'] = round(normalized, 4)
            f_out.write(json.dumps(ex) + "\n")

    print(f"Saved {len(examples)} fully scored examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()