from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_from_disk # type: ignore
import re
from math_agent.utils.commons import save_evaluation_results, run_full_evaluation

prompt = """
Evaluate any defined expressions or constants.
Substitute known values into subsequent equations where applicable.
Solve any resulting systems of equations to find the values of unknowns.
If a variable is defined in terms of another (e.g., x=ay+bx=ay+b), solve for the required variable.
Present all steps clearly and logically, showing how each result is derived from the previous one.
Ensure all algebraic manipulations are valid. Simplify expressions where appropriate.
Provide the final answer, along with intermediate steps as needed for clarity. At the end, give just a numerical or symbolic answer.
At the end, clearly separate the final numerical answer using the format:
FinalAnswer: <value>
"""

def parse_llm_output(output_text):
    """
    Parse LLM output to extract the final answer.
    
    Args:
        output_text (str): The raw output from the LLM
        
    Returns:
        dict: Contains 'answer', 'reasoning', and 'success' fields
    """
    try:
        # Clean the text first
        cleaned_text = output_text.strip()
        
        # Remove Qwen specific tokens if present
        cleaned_text = cleaned_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
        
        # Remove other common tokens
        cleaned_text = cleaned_text.replace("<bos>", "").replace("<end_of_turn>", "").strip()
        
        # If we have chat template sections, extract the assistant's response
        if "<start_of_turn>" in cleaned_text:
            parts = cleaned_text.split("<start_of_turn>")
            if len(parts) > 1:
                assistant_response = parts[-1].strip()
            else:
                assistant_response = cleaned_text
        else:
            assistant_response = cleaned_text
        
        # If the response is empty or too short, return failure
        if not assistant_response or len(assistant_response) < 2:
            return {
                'answer': None,
                'reasoning': output_text,
                'success': False,
                'raw_output': output_text
            }
        
        # Extract the final answer using multiple patterns
        final_answer_patterns = [
            r"FinalAnswer:\s*([^\n<]+)",  # Standard format
            r"Final Answer:\s*([^\n<]+)",  # Alternative format
            r"Answer:\s*([^\n<]+)",  # Simple format
            r"Therefore,?\s*[a-zA-Z]?\s*=\s*([^\n<]+)",  # Pattern like "Therefore, b = -6"
            r"The answer is\s*([^\n<]+)",  # Natural language format
            r"The probability is\s*([^\n<]+)",  # For probability problems
            r"=\s*([^\n<]+?)(?:\s|$)",  # Pattern like "= 0.5"
        ]
        
        extracted_answer = None
        for pattern in final_answer_patterns:
            match = re.search(pattern, assistant_response, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip()
                break
        
        # If no pattern matched, try to find the last meaningful number or expression
        if not extracted_answer:
            # Look for fractions like "1/28" or "0/1" 
            fraction_pattern = r"([0-9]+/[0-9]+)"
            fraction_matches = re.findall(fraction_pattern, assistant_response)
            if fraction_matches:
                extracted_answer = fraction_matches[-1]
            else:
                # Look for decimal numbers
                decimal_pattern = r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                decimal_matches = re.findall(decimal_pattern, assistant_response)
                if decimal_matches:
                    extracted_answer = decimal_matches[-1]
                else:
                    # Look for standalone numbers
                    number_pattern = r"([0-9]+)"
                    number_matches = re.findall(number_pattern, assistant_response)
                    if number_matches:
                        extracted_answer = number_matches[-1]
        
        # Clean up the extracted answer
        if extracted_answer:
            extracted_answer = extracted_answer.strip(".,!?;")
            # Remove any trailing punctuation or markdown
            extracted_answer = re.sub(r'[*_`]+', '', extracted_answer)
            
        return {
            'answer': extracted_answer,
            'reasoning': assistant_response,
            'success': extracted_answer is not None,
            'raw_output': output_text
        }
        
    except Exception as e:
        return {
            'answer': None,
            'reasoning': output_text,
            'success': False,
            'error': str(e),
            'raw_output': output_text
        }

def evaluate_solution(predicted_answer, ground_truth):
    """
    Evaluate if the predicted answer matches the ground truth.
    
    Args:
        predicted_answer (str): The extracted answer from LLM
        ground_truth (str): The correct answer
        
    Returns:
        bool: True if answers match, False otherwise
    """
    if not predicted_answer or not ground_truth:
        return False
    
    # Clean both answers
    def clean_answer(ans):
        if ans is None:
            return None
        # Remove whitespace and convert to string
        ans = str(ans).strip()
        # Remove common formatting
        ans = ans.replace(" ", "").replace(",", "")
        return ans.lower()
    
    pred_clean = clean_answer(predicted_answer)
    truth_clean = clean_answer(ground_truth)
    
    if pred_clean == truth_clean:
        return True
    
    # Try to convert to numbers for comparison
    try:
        pred_num = float(pred_clean)
        truth_num = float(truth_clean)
        # Check if numbers are close (handle floating point precision)
        return abs(pred_num - truth_num) < 1e-6
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return pred_clean == truth_clean

def solve_math_problem(problem, expected_answer=None, model=None, tokenizer=None, device=None):
    """
    Solve a math problem using the LLM and parse the result.
    Compatible with commons.py evaluation functions.
    
    Args:
        problem (str): The math problem to solve
        expected_answer (str, optional): The expected answer for evaluation
        model: The loaded language model
        tokenizer: The tokenizer for the model
        device (str): The device to run the model on
        
    Returns:
        dict: Contains the parsed result and evaluation metrics
    """
    try:
        # Prepare the input with proper chat template
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": problem}]
        
        # Apply chat template and add assistant start token
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print(f"DEBUG: Input text: {input_text}")
        
        # Generate response with adjusted parameters
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                max_new_tokens=28,
                temperature=0.3,
                top_p=0.85,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        
        # Decode only the new tokens (generated part)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Also get the full output for debugging
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        print('DEBUG: GENERATED TEXT: ')
        print(generated_text)
        print('DEBUG: RAW OUTPUT: ')
        print(raw_output)
        print('='*100)
        
        # Parse the output using the generated text
        parsed_result = parse_llm_output(generated_text if generated_text.strip() else raw_output)
        
        # Evaluate if expected answer is provided
        if expected_answer is not None:
            parsed_result['correct'] = evaluate_solution(parsed_result['answer'], expected_answer) if parsed_result['success'] else False
            parsed_result['expected_answer'] = expected_answer
        
        return parsed_result
        
    except Exception as e:
        return {
            'answer': None,
            'success': False,
            'error': str(e),
            'raw_output': f"Error: {str(e)}",
            'correct': False,
            'expected_answer': expected_answer
        }

def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print('USING DEVICE: ', device)

    checkpoint = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    # Fix the pad token issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    print("Starting comprehensive evaluation...")

    dataset = load_from_disk("data/processed/math_qa_dataset")
    
    # Use the common evaluation functions from commons.py
    solve_function = solve_math_problem
    solve_function_args = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }
    
    full_results = run_full_evaluation(dataset, solve_function, solve_function_args, problems_per_category=5, verbose=True)
    save_evaluation_results(full_results, "comprehensive_evaluation_results.json")
    print("Evaluation complete! Results saved to comprehensive_evaluation_results.json")

if __name__ == "__main__":
    main()