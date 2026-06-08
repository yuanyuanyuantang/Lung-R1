#!/usr/bin/env python3
"""KGQA knowledge Q&A evaluation — Two-step method: first answer, then score on a 0-5 scale"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ================= Configuration area =================
# Answer model (7b-full local deployment)
TEST_MODEL_URL = "http://localhost:8080/v1"
TEST_MODEL_KEY = "EMPTY"
TEST_MODEL_NAME = "<your-model-path>"  # Replace with local model path or HuggingFace model ID

# Judge model
JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_API_KEY = "***"  # Replace with your own API Key
JUDGE_BASE_URL = "https://api.deepseek.com"

# Concurrency
PREDICT_WORKERS = 30   # Answer model concurrency (local deployment, should not be too high)
JUDGE_WORKERS = 30     # Judge model concurrency (lower to avoid rate limiting)

# Judge API retry
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_DELAY = 2  # seconds

# File paths
INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_factory_kg.json")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
# ============================================

JUDGE_SYSTEM_PROMPT = """你是一名三甲医院呼吸与危重症医学科主任医师，负责评估 AI 模型对呼吸医学知识问答的回答质量。

你的任务：结合题目原文，将模型的"预测答案"与"参考答案"进行比对，从医学准确性角度进行 0-5 分打分。

【评分标准】

5 分（完全正确）
- 答案全部关键要点均准确覆盖，与参考答案高度一致或医学等价
- 无任何医学性错误或遗漏核心要点

4 分（基本正确）
- 主要关键点已覆盖，可能存在轻微细节缺失或表述不够精确
- 不影响实际医学判断方向
- 不得出现明显医学错误

3 分（部分正确）
- 覆盖部分关键点，但遗漏重要医学内容
- 或概念混淆、逻辑不严谨、表达模糊
- 大方向正确，但不足以作为独立判断依据

2 分（错误较多）
- 仅少部分正确内容
- 大部分关键医学要点缺失
- 方向明显偏离

1 分（严重错误）
- 存在明确医学错误或方向性错误
- 严重缺乏关键医学信息

0 分（完全错误/无关）
- 完全离题、拒绝回答、或完全错误

【重要原则】
1. 以医学正确性为最高优先级
2. 允许同义/近义表达（医学等价即可）
3. 关键信息遗漏要扣分
4. 仔细阅读题目原文，结合上下文理解答案的合理性"""

USER_PROMPT_TEMPLATE = """请评估以下医学知识问答答案：

【题目指令】{instruction}

【题目内容】{input_text}

【参考答案】{ground_truth}
【模型答案】{prediction}

请结合题目原文，对模型答案进行 0-5 分打分。

严格按以下 JSON 格式输出，不要输出其他内容：
{{
  "score": 0-5 的整数,
  "reason": "判定理由（一两句话）"
}}"""


def call_predict(client, model, instruction, input_text, system_prompt, max_retries=2):
    """Call answer model to generate prediction"""
    prompt = f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Answer model call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def call_judge(client, model, instruction, input_text, ground_truth, prediction):
    """Call judge model for evaluation"""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        input_text=input_text,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=300,
            )
            break
        except Exception as e:
            if attempt < JUDGE_MAX_RETRIES - 1:
                delay = JUDGE_RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
            else:
                raise e

    result_text = response.choices[0].message.content.strip()
    # Clean markdown code blocks
    if result_text.startswith("```"):
        lines = result_text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        result_text = "\n".join(lines).strip()
    # Try to extract JSON from the text
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if json_match:
        result_text = json_match.group(0)
    result = json.loads(result_text)
    # Validate score field
    if "score" not in result:
        raise ValueError(f"Missing 'score' field in judge output: {result_text}")
    score = result["score"]
    if not isinstance(score, (int, float)) or score < 0 or score > 5:
        raise ValueError(f"Invalid score value: {score}")
    return result


def judge_one(item, judge_client, judge_model):
    """Judge a single sample"""
    try:
        judgment = call_judge(
            judge_client, judge_model,
            item["instruction"], item["input"], item["ground_truth"], item["prediction"]
        )
        return {
            "index": item.get("index", -1),
            "instruction": item["instruction"],
            "input": item["input"],
            "prediction": item["prediction"],
            "ground_truth": item["ground_truth"],
            "score": judgment["score"],
            "reason": judgment.get("reason", ""),
            "success": True,
        }
    except Exception as e:
        return {
            "index": item.get("index", -1),
            "instruction": item["instruction"],
            "input": item["input"],
            "prediction": item["prediction"],
            "ground_truth": item["ground_truth"],
            "error": str(e),
            "success": False,
        }


def compute_stats(results):
    """Compute evaluation statistics"""
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    failed = total - success

    scores = [r["score"] for r in results if r.get("success") and "score" in r]
    score_dist = {i: 0 for i in range(6)}
    for s in scores:
        if isinstance(s, int) and 0 <= s <= 5:
            score_dist[s] += 1

    stats = {
        "total_samples": total,
        "evaluated_samples": success,
        "failed_samples": failed,
        "score_distribution": score_dist,
        "mean_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "score_5_rate": round(score_dist[5] / len(scores) * 100, 2) if scores else 0,
        "score_4_5_rate": round((score_dist[4] + score_dist[5]) / len(scores) * 100, 2) if scores else 0,
    }
    return stats


def main():
    # Initialize clients
    predict_client = OpenAI(api_key=TEST_MODEL_KEY, base_url=TEST_MODEL_URL, timeout=120)
    judge_client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL, timeout=120)

    # Load data
    print(f"Loading data: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total {len(data)} records")

    # Resume from checkpoint: load existing results
    all_results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                all_results = loaded
            elif isinstance(loaded, dict) and "results" in loaded:
                all_results = loaded["results"]
            print(f"Loaded {len(all_results)} existing results, skipping...")
        except Exception:
            pass

    # Filter already processed samples
    processed_indices = set()
    for r in all_results:
        idx = r.get("index", -1)
        if idx >= 0:
            processed_indices.add(idx)

    remaining = []
    for i, item in enumerate(data):
        if i not in processed_indices:
            item["index"] = i
            remaining.append(item)

    if len(remaining) == 0:
        print("All data processed")
    else:
        print(f"Pending {len(remaining)} records")

        # Step 1: Call answer model to generate predictions
        print(f"\n[Step 1] Calling answer model for predictions (concurrency={PREDICT_WORKERS})...")
        predict_results = {}
        system_prompt = "你是呼吸内科辅助诊断助手。请根据题目内容，给出简明准确的回答，直接回答问题即可。"

        def predict_one(item):
            ans = call_predict(
                predict_client, TEST_MODEL_NAME,
                item["instruction"], item["input"], system_prompt
            )
            return item["index"], ans

        with ThreadPoolExecutor(max_workers=PREDICT_WORKERS) as executor:
            futures = {executor.submit(predict_one, item): item for item in remaining}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                idx, ans = future.result()
                predict_results[idx] = ans
                if done_count % 100 == 0 or done_count == len(remaining):
                    ok = sum(1 for v in predict_results.values() if v is not None)
                    print(f"  Answer progress: {done_count}/{len(remaining)}, success: {ok}")

        # Assemble evaluation data
        eval_items = []
        for item in remaining:
            idx = item["index"]
            pred = predict_results.get(idx)
            if pred is None:
                print(f"  Skipping id={idx}: answer failed")
                continue
            eval_items.append({
                "index": idx,
                "instruction": item["instruction"],
                "input": item["input"],
                "prediction": pred,
                "ground_truth": item["output"],
            })

        # Step 2: Call judge model for evaluation
        print(f"\n[Step 2] Calling judge model for evaluation (concurrency={JUDGE_WORKERS})...")

        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as executor:
            futures = {
                executor.submit(judge_one, item, judge_client, JUDGE_MODEL): item
                for item in eval_items
            }
            done_count = 0
            batch_results = []
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                batch_results.append(result)
                if done_count % 50 == 0 or done_count == len(eval_items):
                    ok = sum(1 for r in batch_results if r["success"])
                    print(f"  Judge progress: {done_count}/{len(eval_items)}, success: {ok}")

        all_results.extend(batch_results)

    # Sort
    all_results.sort(key=lambda x: x.get("index", -1))

    # Save detailed results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Statistics
    print("\n=== Evaluation Result Statistics ===")
    stats = compute_stats(all_results)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # Save final results (with statistics)
    final_output = {**stats, "results": all_results}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"\nFinal results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
