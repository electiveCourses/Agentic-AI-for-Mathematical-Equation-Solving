import csv
import glob
import os
import re
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = "../../data/math_qa"  
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPLITS = ["train-easy", "train-medium", "train-hard", 
          "interpolate", "extrapolate"]


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS backend")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).eval()
    model.to(device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    ).eval()
else:
    device = torch.device("cpu")
    print("Using CPU")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).eval()

def get_llm_answer(question):
    prompt = f"""Question: {question}\nGive only the answer.Your answer can be in various formats:                                                                          │
│ - Single numbers: "5", "3.14159", "-2"                                                                          │
│ - Multiple numbers: "2, 3, 5", "0.5, 0.2, 0.1, -5.0"                                                            │
│ - Fractions: "1/2", "3/4", "7/8"                                                                                │
│ - Symbolic expressions: "x^2 + 3*x + 2", "sqrt(2)", "2*pi"                                                      │
│ - Lists/sets: "[1, 2, 3\]"                                                                         │
│ - Complex expressions: "2 + 3*I", "sqrt(5) + 2"   
    """
    

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(DEVICE)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
   
    answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    

    match = re.search(r"(-?\d+(?:\.\d+)?)", answer)
    return match.group(1) if match else answer.strip()


def process_split(split_name):
    print(f"\n=== Processing {split_name} ===")
    split_dir = os.path.join(DATA_DIR, split_name)
    if not os.path.exists(split_dir):
        print(f"Directory {split_dir} does not exist, skipping...")
        return []

    results = []
    files = sorted(glob.glob(os.path.join(split_dir, "*.txt")))
    print(f"Found {len(files)} files in {split_name}")

    for file in files:
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 2):
            question = lines[i]
            gt_answer = lines[i + 1] if i + 1 < len(lines) else ""
            llm_answer = get_llm_answer(question)
            correct = llm_answer == gt_answer
            results.append({
                "split": split_name,
                "file": os.path.basename(file),
                "question": question,
                "gt_answer": gt_answer,
                "llm_answer": llm_answer,
                "correct": correct,
            })
            print(
                f"Q: {question}\nGT: {gt_answer}\nLLM: {llm_answer}\nCorrect: {correct}\n---"
            )

    return results


all_results = []
split_accuracies = {}

for split in SPLITS:
    split_results = process_split(split)
    all_results.extend(split_results)

    if split_results:
        accuracy = sum(r["correct"] for r in split_results) / len(split_results)
        split_accuracies[split] = accuracy
        print(f"{split} - Total: {len(split_results)} | Accuracy: {accuracy:.3f}")
    else:
        split_accuracies[split] = 0.0
        print(f"{split} - No data found")


results_csv = "../../results/baseline/baseline_results.csv"
with open(results_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["split", "file", "question", "gt_answer", "llm_answer", "correct"],
    )
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)


print("\n=== FINAL RESULTS ===")
print(f"Results saved to: {results_csv}")
print(f"Total problems processed: {len(all_results)}")

if all_results:
    overall_accuracy = sum(r["correct"] for r in all_results) / len(all_results)
    print(f"Overall accuracy: {overall_accuracy:.3f}")

print("\nAccuracy by split:")
for split, accuracy in split_accuracies.items():
    print(f"  {split}: {accuracy:.3f}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def evaluate_all_subfolders(data_root, agent, num_problems=None):
    results_dir = "results/baseline"
    ensure_dir(results_dir)
    subfolders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    print(f"Found subfolders: {subfolders}")

    for subfolder in subfolders:
        subfolder_path = os.path.join(data_root, subfolder)
        print(f"\n=== Evaluating {subfolder_path} ===")
        dataset = load_qa_dataset(subfolder_path)
        if dataset:
            results = run_evaluation(dataset, agent, num_problems=num_problems)
            out_path = os.path.join(results_dir, f"{subfolder}_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {out_path}")
        else:
            print(f"No valid dataset in {subfolder_path}")

if __name__ == "__main__":
    data_root = "data/math_qa"
    evaluate_all_subfolders(data_root, agent) 