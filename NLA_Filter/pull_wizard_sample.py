import json
import random

INPUT_FILE = "wizardlm_converted.jsonl"
OUTPUT_FILE = "wizard_500_sample.jsonl"
SAMPLE_SIZE = 500

def pull_sample():
    print("Reading converted data...")
    all_lines = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            all_lines.append(line)
            
    print(f"Total lines available: {len(all_lines)}")
    
    # Shuffle and slice
    random.seed(42) # Keeps it reproducible
    sampled_lines = random.sample(all_lines, min(SAMPLE_SIZE, len(all_lines)))
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in sampled_lines:
            f_out.write(line)
            
    print(f"Successfully pulled {len(sampled_lines)} random examples into {OUTPUT_FILE}")

if __name__ == "__main__":
    pull_sample()