import ast
import csv
import glob
import os
import re
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Параметры
DATA_DIR = "data/math_qa"  # Базовая директория
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Список всех splits
SPLITS = ["train-easy", "train-medium", "train-hard", "interpolate", "extrapolate"]

# Загрузка модели
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)


def get_llm_code(question, answer_type_hint=None):
    """Получает Python-код от LLM для решения задачи с учетом типа ответа"""
    # Формируем инструкцию в зависимости от типа задачи
    if answer_type_hint == "list":
        format_hint = (
            "Print only the final result as a Python list of numbers, e.g. [1, 2, 3]"
        )
    elif answer_type_hint == "bool":
        format_hint = "Print only the final result as True or False (Python boolean)"
    elif answer_type_hint == "expr":
        format_hint = (
            "Print only the final result as a mathematical expression (e.g. 2*(x+1))"
        )
    else:
        format_hint = "Print only the final result (number, boolean, list, or expression) with no extra text."

    prompt = f"""Question: {question}

Write Python code to solve this mathematical problem. The code should:
1. Calculate the answer
2. {format_hint}

Example:
Question: What is 2 + 3?
Code:
result = 2 + 3
print(result)

Your code:"""

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        inputs, max_new_tokens=200, temperature=0.2, top_p=0.9, do_sample=True
    )
    code_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    code_match = re.search(r"```python\s*(.*?)\s*```", code_response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    else:
        code_start = code_response.find("Your code:")
        if code_start != -1:
            return code_response[code_start + 11 :].strip()
        return code_response.strip()


def parse_output(output):
    """Пытается привести вывод к числу, списку, булеву или оставить строкой"""
    output = output.strip()
    # Попробуем как число
    try:
        if "." in output:
            return float(output)
        return int(output)
    except Exception:
        pass
    # Попробуем как список
    try:
        val = ast.literal_eval(output)
        if isinstance(val, (list, tuple)):
            return list(val)
    except Exception:
        pass
    # Попробуем как булево
    if output in {"True", "False"}:
        return output == "True"
    # Если это выражение (например, 2*(x+1)), оставим строкой
    return output


def execute_code_safely(code):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=tempfile.gettempdir(),
        )
        os.unlink(temp_file)
        if result.returncode == 0:
            output = result.stdout.strip()
            return parse_output(output)
        else:
            return f"ERROR: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout"
    except Exception as e:
        return f"ERROR: {str(e)}"


def guess_answer_type(gt_answer):
    """Грубая эвристика для типа ответа по эталонному ответу"""
    gt = gt_answer.strip()
    if gt in {"True", "False"}:
        return "bool"
    if gt.startswith("[") and gt.endswith("]"):
        return "list"
    if re.match(r"^-?\d+(\.\d+)?$", gt):
        return "number"
    # Матем. выражение: содержит буквы, скобки, операторы
    if re.match(r"^[\w\d\*\+\-/\(\)\^ ]+$", gt) and any(c.isalpha() for c in gt):
        return "expr"
    return None


def compare_answers(pred, gt):
    """Сравнивает предсказание и эталон с учетом типа"""
    if isinstance(gt, (int, float)) and isinstance(pred, (int, float)):
        return abs(float(pred) - float(gt)) < 1e-6
    if isinstance(gt, list) and isinstance(pred, list):
        return [str(x) for x in pred] == [str(x) for x in gt]
    if isinstance(gt, bool) and isinstance(pred, bool):
        return pred == gt
    # Для выражений и строк — сравнение строк
    return str(pred).strip() == str(gt).strip()


def get_llm_answer_with_code(question, gt_answer):
    answer_type_hint = guess_answer_type(gt_answer)
    code = get_llm_code(question, answer_type_hint)
    print(f"Generated code:\n{code}\n")
    result = execute_code_safely(code)
    print(f"Execution result: {result}\n")
    gt_parsed = parse_output(gt_answer)
    correct = compare_answers(result, gt_parsed)
    return result, correct


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
            llm_answer, correct = get_llm_answer_with_code(question, gt_answer)
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
results_csv = "baseline2_code_execution_results.csv"
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
