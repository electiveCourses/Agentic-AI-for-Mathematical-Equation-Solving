import os
import json
from typing import Dict, Any, List
from sympy import sympify, simplify
from fractions import Fraction

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.llms import OpenAI  # Можно заменить на HuggingFaceHub или другой LLM
from langchain_openai import ChatOpenAI
import unicodedata

def to_ascii(s):
    if not isinstance(s, str):
        s = str(s)
    # Удаляет все не-ASCII символы, заменяя их на похожие или убирая
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

# --- API KEY LOADING ---
SECRETS_PATH = os.path.join('secrets', 'openai.key')
def load_openai_key():
    try:
        with open(SECRETS_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        raise RuntimeError(f"OpenAI API key not found in {SECRETS_PATH}. Please create this file and put your key there.")

# --- TOOLS ---

def exact_match_tool(pred: str, gt: str) -> str:
    """Проверка точного совпадения строк"""
    return "CORRECT" if pred.strip() == gt.strip() else "INCORRECT"

def rounded_match_tool(pred: str, gt: str, ndigits: int = 4) -> str:
    """Проверка совпадения с округлением"""
    try:
        return "CORRECT" if round(float(pred), ndigits) == round(float(gt), ndigits) else "INCORRECT"
    except Exception:
        return "ERROR"

def fraction_match_tool(pred: str, gt: str) -> str:
    """Проверка совпадения дробей"""
    try:
        return "CORRECT" if Fraction(pred) == Fraction(gt) else "INCORRECT"
    except Exception:
        return "ERROR"

def symbolic_match_tool(pred: str, gt: str) -> str:
    """Проверка смыслового совпадения через sympy"""
    try:
        pred_expr = sympify(pred)
        gt_expr = sympify(gt)
        return "CORRECT" if simplify(pred_expr - gt_expr) == 0 else "INCORRECT"
    except Exception:
        return "ERROR"

# --- LLM DECISION TOOL ---
def llm_decision_tool(pred: str, gt: str, problem: str, llm) -> str:
    prompt = (
        f"Check if the answer to the math problem is correct.\n"
        f"Problem: {to_ascii(problem)}\n"
        f"Expected: {to_ascii(gt)}\n"
        f"Predicted: {to_ascii(pred)}\n"
        f"Reply only 'CORRECT' or 'INCORRECT' and briefly explain why."
    )
    print(repr(prompt))
    response = llm.invoke([{"role": "user", "content": prompt}])
    result = response.content
    if "CORRECT" in result.upper():
        return "CORRECT"
    return "INCORRECT"

# --- MAIN EVAL AGENT ---
def evaluate_with_agent(results_path: str, llm_api_key: str) -> Dict[str, Any]:
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Если это словарь с ключом 'details', берём только его
    if isinstance(data, dict) and 'details' in data:
        data = data['details']
        print("Используем data['details'] для оценки.")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        print("Файл уже в правильном формате.")
    else:
        print("Неожиданный формат данных:", type(data), data)
        exit(1)

    # Новый способ инициализации LLM
    llm = ChatOpenAI(openai_api_key=llm_api_key, temperature=0.0, model_name="gpt-3.5-turbo")

    # Описываем инструменты для агента
    tools = [
        Tool(name="ExactMatch", func=lambda args: exact_match_tool(args['pred'], args['gt']), description="Точное совпадение"),
        Tool(name="RoundedMatch", func=lambda args: rounded_match_tool(args['pred'], args['gt']), description="Сравнение с округлением"),
        Tool(name="FractionMatch", func=lambda args: fraction_match_tool(args['pred'], args['gt']), description="Сравнение дробей"),
        Tool(name="SymbolicMatch", func=lambda args: symbolic_match_tool(args['pred'], args['gt']), description="Символьное сравнение"),
        Tool(name="LLMDecision", func=lambda args: llm_decision_tool(args['pred'], args['gt'], args['problem'], llm), description="LLM-проверка")
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    details: List[Dict[str, Any]] = []
    total = 0
    correct = 0
    incorrect = 0
    errors = 0

    for item in data:
        pred = item.get('got') or item.get('answer') or item.get('predicted')
        gt = item.get('expected') or item.get('ground_truth') or item.get('answer')
        problem = item.get('problem')
        status = 'incorrect'
        tool_used = None
        error_msg = None

        if pred is None or gt is None:
            status = 'error'
            errors += 1
            error_msg = 'Missing prediction or ground truth'
        else:
            # Пробуем все tools по очереди
            for tool in tools[:-1]:  # кроме LLM
                result = tool.func({'pred': pred, 'gt': gt})
                if result == "CORRECT":
                    status = tool.name
                    correct += 1
                    tool_used = tool.name
                    break
            else:
                # Если ни один tool не сработал, спрашиваем LLM
                result = tools[-1].func({'pred': pred, 'gt': gt, 'problem': problem})
                if result == "CORRECT":
                    status = "LLMDecision"
                    correct += 1
                    tool_used = "LLMDecision"
                else:
                    status = "incorrect"
                    incorrect += 1
                    tool_used = "none"

        total += 1
        details.append({
            'problem': problem,
            'expected': gt,
            'predicted': pred,
            'status': status,
            'tool_used': tool_used,
            'error': error_msg
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        'total': total,
        'correct': correct,
        'incorrect': incorrect,
        'errors': errors,
        'accuracy': accuracy,
        'details': details
    }

if __name__ == '__main__':
    print('=== Math Agent Evaluation (LangChain + LLM) ===')
    openai_key = load_openai_key()
    stats = evaluate_with_agent('results/train_medium_results_fixed.json', openai_key)
    print(f"Total: {stats['total']}")
    print(f"Correct: {stats['correct']}")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Errors: {stats['errors']}")
    print(f"Accuracy: {stats['accuracy']:.2%}")
    with open('results/eval_train_meduim_stats_llm.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print('Detailed stats saved to results/eval_train_meduim_stats_llm.json') 