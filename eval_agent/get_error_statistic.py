#пример для запуска python3 eval_agent/get_error_statistic.py train_
#train_ - начало названий файлов, которые необходимо проверить
import os
import json
import sys
import glob
from collections import defaultdict
from math import isclose

def is_equivalent(a, b, tol=1e-6):
    try:
        return isclose(float(a), float(b), abs_tol=tol)
    except Exception:
        return str(a).strip() == str(b).strip()

def is_digits_after_point_error(a, b, tol=1e-6):
    try:
        fa, fb = float(a), float(b)
        return isclose(fa, fb, abs_tol=tol) and str(a) != str(b)
    except Exception:
        return False

def contains_cuda_oom(item):
    for v in item.values():
        if isinstance(v, str) and "cuda out of memory" in v.lower():
            return True
    return False

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'details' in data:
        data = data['details']
    
    category_stats = defaultdict(lambda: {'total': 0, 'errors': 0, 'cuda_oom': 0, 'too_many_steps': 0, 'equiv_errors': 0, 'digits_error': 0})
    cuda_oom_count = 0
    too_many_steps_count = 0
    digits_error_count = 0
    skipped = 0
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            skipped += 1
            if skipped <= 3:
                print(f"  [warn] Skipping non-dict entry at index {idx}: {repr(item)[:80]}")
            continue
        cat = item.get('category', 'unknown')
        category_stats[cat]['total'] += 1
        pred = str(item.get('answer', item.get('got', ''))).strip()
        gt = str(item.get('expected', '')).strip()
        verdict = str(item.get('verdict', item.get('status', ''))).lower()
        error = str(item.get('error', '')).lower() if 'error' in item else ''
        correct = item.get('correct', None)
        # CUDA OOM (ищем по всем полям)
        if contains_cuda_oom(item):
            category_stats[cat]['cuda_oom'] += 1
            cuda_oom_count += 1
            category_stats[cat]['too_many_steps'] += 1  # CUDA OOM считается too many steps
            too_many_steps_count += 1
        elif 'too many steps' in error or 'too many steps' in verdict:
            category_stats[cat]['too_many_steps'] += 1
            too_many_steps_count += 1
        # Ошибка только в количестве знаков после точки
        digits_error = False
        if pred and gt and is_digits_after_point_error(pred, gt):
            # Если явно указано correct==False или verdict не correct, или verdict отсутствует
            if (correct is False) or (verdict and verdict not in ['correct', 'no_answer', 'correct_by_gemini']) or (verdict == '' and correct is False):
                category_stats[cat]['digits_error'] += 1
                digits_error_count += 1
                digits_error = True
        # Ошибка по эквивалентности (но не digits_error)
        if not digits_error:
            if (verdict not in ['correct', 'no_answer', 'correct_by_gemini']) and not is_equivalent(pred, gt):
                category_stats[cat]['equiv_errors'] += 1
                category_stats[cat]['errors'] += 1
    if skipped:
        print(f"  [info] Skipped {skipped} non-dict entries in {os.path.basename(filepath)}.")
    return category_stats, cuda_oom_count, too_many_steps_count, digits_error_count

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 batch_results_analyzer.py <file_prefix | all>")
        print("Example: python3 batch_results_analyzer.py train_")
        print("         python3 batch_results_analyzer.py math_eval_")
        print("         python3 batch_results_analyzer.py all")
        sys.exit(1)
    prefix = sys.argv[1]
    results_dir = 'results'
    pattern = os.path.join(results_dir, f'{prefix}*.json') if prefix != 'all' else os.path.join(results_dir, '*.json')
    files = glob.glob(pattern)
    if not files:
        print(f'No files found for pattern: {pattern}')
        return
    for filepath in files:
        print(f'\n=== Analyzing {os.path.basename(filepath)} ===')
        stats, cuda_oom, too_many_steps, digits_error = analyze_file(filepath)
        total = sum(cat['total'] for cat in stats.values())
        total_errors = sum(cat['errors'] for cat in stats.values())
        total_equiv_errors = sum(cat['equiv_errors'] for cat in stats.values())
        total_digits_error = sum(cat['digits_error'] for cat in stats.values())
        print(f'Total problems: {total}')
        print(f'Total errors (not equivalent): {total_errors}')
        print(f'Total CUDA OOM errors: {cuda_oom}')
        print(f'Total "too many steps" errors: {too_many_steps}')
        print(f'Number of digits after point errors: {digits_error}')
        print('\nPer-category stats:')
        for cat, s in stats.items():
            print(f'  {cat}: total={s["total"]}, errors={s["errors"]}, cuda_oom={s["cuda_oom"]}, too_many_steps={s["too_many_steps"]}, equiv_errors={s["equiv_errors"]}, digits_error={s["digits_error"]}')

if __name__ == "__main__":
    main() 