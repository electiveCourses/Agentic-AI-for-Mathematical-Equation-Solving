import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class BatchGeminiEvaluator:
    def __init__(self, batch_size: int = 10):
        """Initialize evaluator with proper API key handling and model setup"""
        try:
            # Load API key with multiple fallback options
            self.api_key = self._load_api_key()
            os.environ["GOOGLE_API_KEY"] = self.api_key
            
            # Initialize with standard production model
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.0,
                max_retries=3,
                timeout=30
            )
            
            # Evaluation statistics
            self.batch_size = batch_size
            self.stats = {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "no_answer": 0,
                "api_errors": 0,
                "details": []
            }
            
            # Verify API connectivity before proceeding
            if not self._verify_api_connection():
                raise ConnectionError("API connection verification failed")
                
        except Exception as e:
            raise RuntimeError(f"Evaluator initialization failed: {str(e)}")

    def _load_api_key(self) -> str:
        """Load API key from multiple potential sources"""
        # 1. Check environment variable
        if api_key := os.getenv("GEMINI_API_KEY"):
            return api_key
            
        # 2. Check secrets file
        try:
            with open('secrets/gemini.key', 'r') as f:
                if key := f.read().strip():
                    return key
        except FileNotFoundError:
            pass
            
        # 3. Check alternate file locations
        for path in ['~/.gemini/key', '/etc/gemini/key']:
            try:
                with open(os.path.expanduser(path), 'r') as f:
                    if key := f.read().strip():
                        return key
            except (FileNotFoundError, PermissionError):
                continue
                
        raise ValueError("No valid API key found. Please set GEMINI_API_KEY environment variable "
                        "or create secrets/gemini.key file")

    def _verify_api_connection(self) -> bool:
        """Perform comprehensive API connection check"""
        test_cases = [
            ("What is 1+1? Answer only with number.", "2"),
            ("Is 5 greater than 3? Answer YES or NO.", "YES"),
            ("Capital of France? Answer only with city name.", "Paris")
        ]
        
        for prompt, expected in test_cases:
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                if response.content.strip() != expected:
                    print(f"API test failed. Expected '{expected}', got '{response.content}'")
                    return False
            except Exception as e:
                print(f"API connection test failed with error: {str(e)}")
                return False
                
        return True

    def create_batch_prompt(self, problems: List[Dict]) -> str:
        """Generate optimized batch prompt with clear formatting"""
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
                f"{i}. Problem: {problem}\n"
                f"Expected: {expected}\n"
                f"Answer: {answer}\n"
            )
        
        return "\n".join(prompt_lines)

    def process_batch_response(self, response: str, batch: List[Dict]) -> List[Optional[bool]]:
        """Robust response processing with error handling"""
        results = []
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        for i in range(len(batch)):
            if i < len(lines):
                line = lines[i]
                # Handle both "1. CORRECT" and "CORRECT" formats
                verdict = line.split('.')[-1].strip().upper() if '.' in line else line.upper()
                results.append(verdict == "CORRECT" if verdict in ["CORRECT", "INCORRECT"] else None)
            else:
                results.append(None)
                
        return results

    def evaluate_batch(self, batch: List[Dict]) -> List[Optional[bool]]:
        """Evaluate a batch with comprehensive error handling"""
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
        """Process dataset with progress tracking and adaptive rate limiting"""
        try:
            # Load and validate data
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            problems = data['details'] if isinstance(data, dict) and 'details' in data else data
            
            total_items = len(problems)
            print(f"Starting evaluation of {total_items} problems in batches of {self.batch_size}")
            
            for i in range(0, total_items, self.batch_size):
                batch = problems[i:i + self.batch_size]
                batch_results = self.evaluate_batch(batch)
                
                # Process results
                for item, result in zip(batch, batch_results):
                    self._record_result(item, result)
                
                # Progress reporting
                processed = min(i + self.batch_size, total_items)
                print(f"Progress: {processed}/{total_items} | "
                      f"Accuracy: {self._current_accuracy():.1f}% | "
                      f"API Errors: {self.stats['api_errors']}")
                
                # Adaptive rate limiting
                time.sleep(self._calculate_delay())
                
        except Exception as e:
            raise RuntimeError(f"Dataset evaluation failed: {str(e)}")

    def _record_result(self, item: Dict, result: Optional[bool]):
        """Record individual result with validation"""
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
        """Calculate current accuracy percentage"""
        return (self.stats["correct"] / self.stats["total"] * 100) if self.stats["total"] > 0 else 0.0

    def _calculate_delay(self) -> float:
        """Calculate adaptive delay between batches"""
        base_delay = 2.0  # seconds
        error_rate = self.stats["api_errors"] / self.stats["total"] if self.stats["total"] > 0 else 0
        return min(5.0, max(1.0, base_delay * (1 + error_rate * 3)))

    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        accuracy = self._current_accuracy()
        
        report = [
            "\n=== Math Evaluation Report ===",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: gemini-1.5-flash-latest",
            f"API Key: {self.api_key[:4]}...{self.api_key[-4:]}",
            f"Batch Size: {self.batch_size}",
            f"Total Problems: {self.stats['total']}",
            f"Correct Answers: {self.stats['correct']} ({accuracy:.1f}%)",
            f"Incorrect Answers: {self.stats['incorrect']}",
            f"Missing Answers: {self.stats['no_answer']}",
            f"API Errors: {self.stats['api_errors']}",
            "================================="
        ]
        return "\n".join(report)

    def save_results(self, filename: str = "math_evaluation_results.json"):
        """Save results with complete metadata"""
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
        
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    try:
        print("=== Math Answer Evaluation System ===")
        
        # Initialize with interactive configuration
        batch_size = int(input("Enter batch size (10-50 recommended): ") or "20")
        evaluator = BatchGeminiEvaluator(batch_size=batch_size)
        
        # Run test evaluation
        print("\nRunning connection test...")
        test_results = evaluator.evaluate_batch([
            {"problem": "2+2", "expected": "4", "answer": "4"},
            {"problem": "3*3", "expected": "9", "answer": "9"},
            {"problem": "1/0", "expected": "undefined", "answer": "0"}
        ])
        print(f"Test results: {test_results}")
        
        # Full evaluation
        data_file = input("\nEnter path to evaluation data (default: results/extrapolate_results.json): ") or "results/extrapolate_results.json"
        print(f"\nStarting evaluation of {data_file}...")
        
        start_time = time.time()
        evaluator.evaluate_dataset(data_file)
        
        # Generate and display report
        print("\n" + evaluator.generate_report())
        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        
        # Save results
        evaluator.save_results()
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if 'evaluator' in locals():
            evaluator.save_results("error_results.json")
        exit(1)