import json
import time
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class BatchGeminiEvaluator:
    def __init__(self, batch_size: int = 20):
        try:
            self.api_key = self._load_api_key()
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.0,
                max_retries=3,
                timeout=30
            )
            self.batch_size = batch_size
            self.stats = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "no_answer": 0,
                "api_errors": 0,
                "details": []
            }
        except Exception as e:
            raise RuntimeError(f"Evaluator initialization failed: {str(e)}")

    def _load_api_key(self) -> str:
        if api_key := os.getenv("GEMINI_API_KEY"):
            return api_key
        try:
            with open('secrets/gemini.key', 'r') as f:
                if key := f.read().strip():
                    return key
        except FileNotFoundError:
            pass
        for path in ['~/.gemini/key', '/etc/gemini/key']:
            try:
                with open(os.path.expanduser(path), 'r') as f:
                    if key := f.read().strip():
                        return key
            except (FileNotFoundError, PermissionError):
                continue
        raise ValueError("No valid API key found. Please set GEMINI_API_KEY environment variable or create secrets/gemini.key file")

    def create_batch_prompt(self, problems):
        prompt_lines = [
            "You are a math expert evaluating answer correctness.",
            "For each problem, respond ONLY with:",
            "1. CORRECT or INCORRECT",
            "2. CORRECT or INCORRECT",
            "...\n\nProblems:\n"
        ]
        for i, item in enumerate(problems, 1):
            problem = item.get("problem", "").strip()
            expected = str(item.get("expected", "")).strip()
            answer = str(item.get("answer", "")).strip()
            prompt_lines.append(
                f"{i}. Problem: {problem}\nExpected: {expected}\nAnswer: {answer}\n"
            )
        return "\n".join(prompt_lines)

    def process_batch_response(self, response: str, batch):
        results = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        for i in range(len(batch)):
            if i < len(lines):
                line = lines[i]
                verdict = line.split('.')[-1].strip().upper() if '.' in line else line.upper()
                results.append(verdict == "CORRECT" if verdict in ["CORRECT", "INCORRECT"] else None)
            else:
                results.append(None)
        return results

    def evaluate_batch(self, batch):
        try:
            prompt = self.create_batch_prompt(batch)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            if not response or not response.content:
                raise ValueError("Empty API response")
            return self.process_batch_response(response.content, batch)
        except Exception as e:
            print(f"Batch evaluation failed: {str(e)}")
            self.stats["api_errors"] += 1
            return [None] * len(batch)

    def evaluate_dataset(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        problems = data['details'] if isinstance(data, dict) and 'details' in data else data
        total_items = len(problems)
        print(f"Starting evaluation of {total_items} problems in batches of {self.batch_size}")
        for i in range(0, total_items, self.batch_size):
            batch = problems[i:i + self.batch_size]
            batch_results = self.evaluate_batch(batch)
            for item, result in zip(batch, batch_results):
                self._record_result(item, result)
            processed = min(i + self.batch_size, total_items)
            print(f"Progress: {processed}/{total_items} | Accuracy: {self._current_accuracy():.1f}% | API Errors: {self.stats['api_errors']}")
            time.sleep(1.5)

    def _record_result(self, item, result):
        pred = str(item.get("answer", "")).strip()
        gt = str(item.get("expected", "")).strip()
        if not pred or not gt:
            verdict = "NO_ANSWER"
            is_correct = False
            self.stats["no_answer"] += 1
        elif result is None:
            verdict = "API_ERROR"
            is_correct = False
        else:
            verdict = "CORRECT" if result else "INCORRECT"
            is_correct = result
        self.stats["total"] += 1
        self.stats["correct" if is_correct else "incorrect"] += 1
        self.stats["details"].append({
            "problem": item.get("problem", "").strip(),
            "category": item.get("category", "unknown"),
            "expected": gt,
            "predicted": pred,
            "verdict": verdict,
            "timestamp": datetime.now().isoformat()
        })

    def _current_accuracy(self) -> float:
        return (self.stats["correct"] / self.stats["total"] * 100) if self.stats["total"] > 0 else 0.0

    def save_results(self, filename: str):
        results = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "model": "gemini-1.5-flash-latest",
                "api_key_fingerprint": f"{self.api_key[:4]}...{self.api_key[-4:]}",
                "batch_size": self.batch_size,
                "environment": {
                    "python_version": os.sys.version,
                    "platform": os.sys.platform
                }
            },
            "statistics": self.stats,
            "details": self.stats["details"]
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    print("=== FB Gemini Evaluation ===")
    evaluator = BatchGeminiEvaluator(batch_size=20)
    input_path = "results/agent_comprehensive_evaluation_results.json"
    output_path = "results/code_eval/fb_gemini_eval.json"
    evaluator.evaluate_dataset(input_path)
    evaluator.save_results(output_path) 