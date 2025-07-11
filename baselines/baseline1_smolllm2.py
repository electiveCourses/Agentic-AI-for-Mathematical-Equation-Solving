import csv
import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Параметры
DATA_DIR = "../data/math_qa"  # Базовая директория
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Список всех splits
SPLITS = ["train-easy", "train-medium", "train-hard", "interpolate", "extrapolate"]

# Загрузка модели
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)


# Функция для получения ответа от LLM
def get_llm_answer(question):
    prompt = f"Question: {question}\nGive only the numeric answer."
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs, max_new_tokens=20, temperature=0.2, top_p=0.9, do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только число из ответа
    import re

    match = re.search(r"(-?\d+(?:\.\d+)?)", answer)
    return match.group(1) if match else answer.strip()


# Функция для обработки одного split
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
            results.append(
                {
                    "split": split_name,
                    "file": os.path.basename(file),
                    "question": question,
                    "gt_answer": gt_answer,
                    "llm_answer": llm_answer,
                    "correct": correct,
                }
            )
            print(
                f"Q: {question}\nGT: {gt_answer}\nLLM: {llm_answer}\nCorrect: {correct}\n---"
            )

    return results


# Обработка всех splits
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

# Сохраняем результаты
results_csv = "../results/baseline1_smolllm2_all_splits_results.csv"
with open(results_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=["split", "file", "question", "gt_answer", "llm_answer", "correct"],
    )
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

# Итоговая статистика
print("\n=== FINAL RESULTS ===")
print(f"Results saved to: {results_csv}")
print(f"Total problems processed: {len(all_results)}")

if all_results:
    overall_accuracy = sum(r["correct"] for r in all_results) / len(all_results)
    print(f"Overall accuracy: {overall_accuracy:.3f}")

print("\nAccuracy by split:")
for split, accuracy in split_accuracies.items():
    print(f"  {split}: {accuracy:.3f}")
