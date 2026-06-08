#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LungBench medical question evaluation script.

Usage:
    python benchmark_eval.py --model_name /path/to/model --base_url http://localhost:8080/v1
    python benchmark_eval.py --model_name Qwen/Qwen2.5-7B-Instruct --api_key sk-xxx
    python benchmark_eval.py --model_name /path/to/model --sample_size 50
"""

import os
import json
import time
import re
import argparse
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI


class MedicalQuestionEvaluator:
    def __init__(self, model_name, base_url="http://localhost:8080/v1", api_key="EMPTY"):
        """Initialize model connection and configuration"""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0
        )
        self.model_name = model_name
        
        
    def load_data(self, questions_file, answers_file):
        """Load question and answer data"""
        try:
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers = json.load(f)
            
            # Create answer dictionary for quick lookup
            answer_dict = {item['id']: item['answer'] for item in answers}

            return questions, answer_dict
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], {}
    
    def format_question(self, question_item):
        """Format question as prompt"""
        question = question_item['question']
        options = question_item['option']
        question_type = question_item.get('question_type', '')

        # Filter empty options and build options string
        valid_options = {k: v for k, v in options.items() if v.strip()}
        options_str = ""
        for key, value in valid_options.items():
            options_str += f"{key}. {value}\n"

        # Clarify single choice or multiple choice based on question type
        is_single = question_type == "单项选择题"
        type_hint = "单选题" if is_single else "多选题"
        count_hint = "只需选择一个" if is_single else "请选择所有正确选项"

        prompt = f"""请回答以下医学{type_hint}。

【题目】{question}
【选项】
{options_str}
【要求】{count_hint}。只输出选项字母（如"A"或"AC"），不要输出任何其他文字。必须使用题目中的选项字母A、B、C、D、E、F作答。"""

        return prompt
    
    def extract_answer(self, response_text, question_type=''):
        """Extract answer options from model response

        Args:
            response_text: Raw model response
            question_type: Question type, '单项选择题' or '多项选择题'
        """
        if not response_text or not response_text.strip():
            return "无法识别"

        text = response_text.strip().strip('`').strip()

        # 1. Try to extract pure option letters (e.g., "A" or "AC")
        match = re.match(r'^[A-Fa-f]+$', text)
        if match:
            letters = match.group(0).upper()
            if question_type == '单项选择题' and len(letters) > 1:
                return letters[0]
            return "".join(sorted(set(letters)))

        # 2. Fallback: extract all A-F letters from text
        matches = re.findall(r'[A-Fa-f]', text)
        if matches:
            result = "".join(sorted(set(m.upper() for m in matches)))
            if question_type == '单项选择题' and len(result) > 1:
                return result[0]
            return result

        return "无法识别"

    def get_model_answer(self, prompt):
        """Get single model answer"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个医学专家。请只输出选项字母，不要输出任何其他内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling model: {e}")
            return "模型调用失败"
    
    def get_batch_answers(self, prompts, batch_size=8, max_workers=4):
        """Batch get model answers

        Args:
            prompts: List of prompts, each element is a tuple (question_id, prompt)
            batch_size: Number of questions per batch
            max_workers: Maximum number of concurrent worker threads

        Returns:
            dict: {question_id: model_answer}
        """
        results = {}
        
        def process_prompt(question_id, prompt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个医学专家。请只输出选项字母，不要输出任何其他内容。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                return question_id, response.choices[0].message.content
            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                return question_id, "模型调用失败"

        # Batch processing
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batch processing progress"):
            batch = prompts[i:i+batch_size]

            # Use thread pool for concurrent processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
                future_to_question = {
                    executor.submit(process_prompt, qid, prompt): qid
                    for qid, prompt in batch
                }

                for future in concurrent.futures.as_completed(future_to_question):
                    try:
                        qid, response = future.result()
                        results[qid] = response
                    except Exception as e:
                        print(f"Error getting question result: {e}")

            # Avoid excessive requests
            if i + batch_size < len(prompts):
                time.sleep(0.5)  # Batch interval

        return results
    
    def llm_judge_correct(self, question, correct_answer, model_response, question_type=''):
        """Use LLM to judge if model answer is correct

        Args:
            question: Question content
            correct_answer: Standard answer
            model_response: Raw model response
            question_type: Question type

        Returns:
            bool or None: True=correct, False=incorrect, None=unable to determine
        """
        type_hint = "单选题" if question_type == "单项选择题" else "多选题"
        prompt = f"""请判断以下模型回答是否正确。只输出 true 或 false，不要输出其他任何内容。

【题型】{type_hint}
【题目】{question}
【标准答案】{correct_answer}
【模型回答】{model_response}

请判断模型回答是否正确，只输出 true 或 false。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个判卷助手。只输出 true 或 false。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            result = response.choices[0].message.content.strip().lower()
            if result == "true":
                return True
            elif result == "false":
                return False
            else:
                print(f"LLM judge returned abnormal value: {result}")
                return None
        except Exception as e:
            print(f"LLM judge error: {e}")
            return None

    def evaluate_accuracy(self, questions, answer_dict, sample_size=None, batch_size=8, max_workers=4):
        """Evaluate model accuracy (supports batch processing)

        Args:
            questions: Question list
            answer_dict: Answer dictionary
            sample_size: Sample size
            batch_size: Number of questions per batch
            max_workers: Maximum number of concurrent worker threads

        Returns:
            tuple: (accuracy, partial_accuracy, detailed_results)
        """
        # If sample size is specified, only evaluate a subset of questions
        if sample_size and sample_size < len(questions):
            questions = questions[:sample_size]

        correct_count = 0
        partial_correct_count = 0
        results = []

        # Prepare questions to process
        valid_questions = []
        for question_item in questions:
            question_id = question_item['id']

            # Check if standard answer exists
            if question_id not in answer_dict:
                print(f"No standard answer found for question ID {question_id}")
                results.append({
                    'id': question_id,
                    'correct_answer': '未找到',
                    'model_answer': '未评估',
                    'is_correct': False,
                    'is_partial_correct': False
                })
                continue

            valid_questions.append(question_item)

        if not valid_questions:
            return 0.0, 0.0, results

        print(f"Valid question count: {len(valid_questions)}")
        print(f"Using batch processing, batch size: {batch_size}, concurrency: {max_workers}")

        # Prepare batch request data
        prompts = [(q['id'], self.format_question(q)) for q in valid_questions]

        # Batch get model answers
        start_time = time.time()
        batch_results = self.get_batch_answers(prompts, batch_size, max_workers)
        end_time = time.time()

        print(f"Batch processing complete, time elapsed: {end_time - start_time:.2f}s")

        # Process results
        questions_dict = {q['id']: q for q in valid_questions}

        for question_id, model_response in batch_results.items():
            if question_id in questions_dict:
                question_item = questions_dict[question_id]
                correct_answer = answer_dict[question_id]
                question = question_item['question']
                question_type = question_item.get('question_type', '')

                # Use LLM to judge correctness
                is_correct = self.llm_judge_correct(
                    question, correct_answer, model_response, question_type
                )

                # If False, further use LLM to judge if partially correct (multiple choice)
                is_partial_correct = False
                if is_correct is False and question_type == "多项选择题":
                    is_partial_correct = self.llm_judge_correct(
                        question, correct_answer, model_response + "\n\n补充说明：该回答虽然不完全正确，但是否选择了部分正确答案（没有选错，只是选少了）？如果是部分正确请输出 true，否则输出 false。", question_type
                    )

                if is_correct:
                    correct_count += 1
                elif is_partial_correct:
                    partial_correct_count += 1

                # Save results
                results.append({
                    'id': question_id,
                    'correct_answer': correct_answer,
                    'model_answer': model_response,
                    'is_correct': is_correct if is_correct is not None else False,
                    'is_partial_correct': is_partial_correct if is_partial_correct is not None else False,
                    'question_type': question_type,
                    'exam_type': question_item.get('exam_type', '未知')
                })

        # Calculate accuracy
        if len(results) > 0:
            accuracy = correct_count / len(results) * 100
            partial_accuracy = partial_correct_count / len(results) * 100
        else:
            accuracy = 0
            partial_accuracy = 0

        return accuracy, partial_accuracy, results




    def save_results(self, results, accuracy, partial_accuracy, output_file):
        """Save evaluation results"""
        summary = {
            'total_questions': len(results),
            'correct_answers': sum(1 for r in results if r['is_correct']),
            'partial_correct_answers': sum(1 for r in results if r['is_partial_correct']),
            'accuracy': accuracy,
            'partial_accuracy': partial_accuracy,
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_data = {
            'summary': summary,
            'detailed_results': results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Evaluation results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LungBench medical question evaluation")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--base_url", type=str, default="http://localhost:8080/v1",
                        help="OpenAI-compatible API base URL (default: http://localhost:8080/v1)")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key (default: EMPTY)")
    parser.add_argument("--questions_file", type=str, default=None,
                        help="Question JSON file path (default: questions.json)")
    parser.add_argument("--answers_file", type=str, default=None,
                        help="Answer JSON file path (default: answers.json)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Evaluation result output path (default: scores/model-<name>.json)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to evaluate (default: all questions)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of questions per batch (default: 10)")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum number of concurrent worker threads (default: 10)")
    args = parser.parse_args()

    # Configure file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_file = args.questions_file or os.path.join(script_dir, 'questions.json')
    answers_file = args.answers_file or os.path.join(script_dir, 'answers.json')

    if args.output_file:
        output_file = args.output_file
    else:
        model_short = os.path.basename(args.model_name.rstrip('/'))
        output_file = os.path.join(script_dir, 'scores', f'{model_short}.json')

    # Initialize evaluator
    evaluator = MedicalQuestionEvaluator(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print("Loading data...")
    questions, answer_dict = evaluator.load_data(questions_file, answers_file)

    if not questions or not answer_dict:
        print("Data loading failed, exiting")
        return

    print(f"Successfully loaded {len(questions)} questions")

    sample_size = args.sample_size

    print(f"Starting model accuracy evaluation ({'all questions' if sample_size is None else sample_size} questions)...")
    print(f"Batch processing params: batch_size={args.batch_size}, max_workers={args.max_workers}")

    start_time = time.time()
    accuracy, partial_accuracy, results = evaluator.evaluate_accuracy(
        questions,
        answer_dict,
        sample_size,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    total_time = time.time() - start_time

    print(f"\nEvaluation complete!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per question: {total_time / len(results):.2f}s/question")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct count: {sum(1 for r in results if r['is_correct'])}")
    print(f"Total questions: {len(results)}")

    # Accuracy statistics by question type
    type_stats = {}
    for result in results:
        q_type = result.get('question_type', '未知')
        if q_type not in type_stats:
            type_stats[q_type] = {'total': 0, 'correct': 0}
        type_stats[q_type]['total'] += 1
        if result['is_correct']:
            type_stats[q_type]['correct'] += 1

    print("\nAccuracy statistics by question type:")
    for q_type, stats in type_stats.items():
        type_accuracy = stats['correct'] / stats['total'] * 100
        print(f"{q_type}: {type_accuracy:.2f}% ({stats['correct']}/{stats['total']})")

    # Save results
    evaluator.save_results(results, accuracy, partial_accuracy, output_file)

if __name__ == "__main__":
    main()