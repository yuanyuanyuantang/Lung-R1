import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# ================= Configuration area (modify directly here) =================

# 1. File path configuration
# EMR test data is available upon request to the authors
INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emr_test_data_300.json")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
LIMIT_NUM = None  # Set the number of test cases, None means run all data

# 2. Answer model configuration (The model being tested)
TEST_MODEL_API_URL = "http://localhost:8080/v1"
TEST_MODEL_API_KEY = "EMPTY"
TEST_MODEL_NAME = "<your-model-path>"  # Replace with local model path or HuggingFace model ID

# 3. Judge model configuration (The Judge)
JUDGE_MODEL_API_URL = "https://api.deepseek.com"
JUDGE_MODEL_API_KEY = "***"  # Replace with your own API Key
JUDGE_MODEL_NAME = "deepseek-v4-flash"


# 4. Concurrency control
CONCURRENCY = 10

# 5. Judge model system prompt (evaluates diagnosis only, does not involve treatment/safety)
JUDGE_SYSTEM_PROMPT = """
你是一名三甲医院呼吸与危重症医学科主任医师。
现在需要你对"AI 助手"给出的肺部疾病诊断进行专业打分评估。

【任务说明】
- 你会看到一个完整的病例资料（主诉、现病史、既往史、检查结果等）。
- 你会看到「参考答案」：这是真实临床医生的诊断，格式如 [诊断1],[诊断2]。
- 你会看到「模型答案」：这是 AI 模型给出的诊断，格式类似。
- 你的任务是：对模型答案相对于参考答案的诊断质量进行打分。

【重要原则】
1. 以临床正确性为最高优先级。
2. 允许模型答案与参考答案在措辞上存在合理同义表达（如"肺部感染"vs"肺炎"），只要在医学上等价。
3. 关键诊断遗漏或明显误诊要扣分。
4. 仅根据提供的病例信息进行判断。

【评分维度与标准】（每项 0–5 分）
1. 诊断正确性（diagnosis_score）
   - 5分：主要诊断全部正确识别，无明显误诊。
   - 4分：主要诊断基本正确，部分次要诊断遗漏或表述差异。
   - 3分：诊断方向基本正确，但遗漏多个重要诊断，或存在轻度误诊。
   - 2分：诊断方向有较明显问题，仅有部分正确线索。
   - 1分：大部分诊断错误。
   - 0分：完全错误。

2. 综合分（overall_score）
   - 综合诊断准确性整体主观评价，0–5 分，可以是小数。

【输出要求】
- 必须输出 JSON，且只能输出 JSON，不要附加任何自然语言说明。
- JSON 字段包括：
  - "diagnosis_score": 浮点数，0–5
  - "overall_score": 浮点数，0–5
  - "major_errors": 列表，概括性描述模型答案中最严重的 1–3 个误诊/遗漏
  - "missing_key_points": 列表，列出参考答案中有而模型答案缺失的关键诊断
  - "comments": 字符串，简短总体点评
"""

# =============================================================


def parse_diagnoses(text):
    """Extract diagnosis list from [diagnosis1],[diagnosis2] format"""
    diagnoses = re.findall(r'\[([^\]]+)\]', text)
    return set(d.strip() for d in diagnoses if d.strip())


def compute_diagnosis_metrics(pred_text, ref_text):
    """Compute deterministic diagnosis metrics: exact match, precision, recall, F1"""
    pred = parse_diagnoses(pred_text)
    ref = parse_diagnoses(ref_text)

    if not pred and not ref:
        return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "pred_diagnoses": [], "ref_diagnoses": []}

    if not pred:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "pred_diagnoses": list(pred), "ref_diagnoses": list(ref)}

    if not ref:
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "pred_diagnoses": list(pred), "ref_diagnoses": list(ref)}

    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred == ref else 0.0

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_diagnoses": list(pred),
        "ref_diagnoses": list(ref),
    }


def call_llm_api(api_url, api_key, model_name, messages, temperature=0.0, max_tokens=1024):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if not api_url.endswith("/chat/completions"):
        if api_url.endswith("/"):
            api_url = f"{api_url}chat/completions"
        else:
            api_url = f"{api_url}/chat/completions"

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling API ({model_name}): {e}")
        return None


def get_model_answer(instruction, input_text, system_prompt):
    prompt = f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return call_llm_api(TEST_MODEL_API_URL, TEST_MODEL_API_KEY, TEST_MODEL_NAME, messages)


def evaluate_answer(instruction, input_text, standard_answer, model_answer):
    """
    Evaluate model diagnosis, using the reference answer as the gold standard.
    """
    judge_user_prompt = f"""【病例信息】
{instruction}
{input_text}

【参考答案（真实医生诊断）】
{standard_answer}

【模型答案】
{model_answer}

请根据前述评分标准，对模型答案进行打分并输出 JSON。
"""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt}
    ]
    return call_llm_api(JUDGE_MODEL_API_URL, JUDGE_MODEL_API_KEY, JUDGE_MODEL_NAME, messages)


def process_single_case(case_data):
    system_prompt = case_data.get("system", "你是呼吸内科辅助诊断模型。请根据病例内容，输出仅与肺部相关的疾病诊断，格式为[诊断1],[诊断2],...。不要输出推理过程和治疗建议。")

    # 1. Get model answer
    model_ans = get_model_answer(case_data.get("instruction", ""), case_data.get("input", ""), system_prompt)
    if model_ans is None:
        return None

    standard_ans = case_data.get("output", "")

    # 2. Deterministic diagnosis metrics
    metrics = compute_diagnosis_metrics(model_ans, standard_ans)

    # 3. LLM scoring
    evaluation_str = evaluate_answer(
        case_data.get("instruction", ""),
        case_data.get("input", ""),
        standard_ans,
        model_ans
    )

    try:
        if not evaluation_str:
            raise ValueError("Empty judge output")
        s = evaluation_str.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1]
            if s.endswith("```"):
                s = s.rsplit("\n", 1)[0]
        evaluation = json.loads(s)
    except Exception:
        evaluation = {"raw_output": evaluation_str, "error": "Failed to parse JSON"}

    result = {
        "custom_id": case_data.get("custom_id"),
        "instruction": case_data.get("instruction"),
        "input": case_data.get("input"),
        "standard_output": standard_ans,
        "model_output": model_ans,
        "diagnosis_metrics": metrics,
        "evaluation": evaluation
    }
    return result


def main():
    print(f"Starting evaluation...")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_FILE}")
        return

    if LIMIT_NUM is not None:
        data = data[:LIMIT_NUM]
        print(f"Limiting to first {LIMIT_NUM} cases.")

    results = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                processed_ids = {item['custom_id'] for item in existing_results}
                results = existing_results
                data = [item for item in data if item['custom_id'] not in processed_ids]
                print(f"Resuming... {len(results)} cases already processed. {len(data)} remaining.")
        except json.JSONDecodeError:
            print("Output file exists but is not valid JSON. Starting over.")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        future_to_case = {executor.submit(process_single_case, case): case for case in data}

        for future in tqdm(as_completed(future_to_case), total=len(data), desc="Processing Cases"):
            case = future_to_case[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    if len(results) % 5 == 0:
                        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Exception processing case {case.get('custom_id')}: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done! Results saved to {OUTPUT_FILE}")

    # ====== Print statistics summary ======

    # 1. Deterministic metrics
    total_exact = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0
    for result in results:
        m = result.get("diagnosis_metrics", {})
        if isinstance(m, dict) and "f1" in m:
            total_exact += m.get("exact_match", 0.0)
            total_precision += m.get("precision", 0.0)
            total_recall += m.get("recall", 0.0)
            total_f1 += m.get("f1", 0.0)
            count += 1

    print("\n========== Diagnosis Metrics (Deterministic) ==========")
    if count > 0:
        print(f"  Exact Match  : {total_exact / count:.4f}")
        print(f"  Precision    : {total_precision / count:.4f}")
        print(f"  Recall       : {total_recall / count:.4f}")
        print(f"  F1           : {total_f1 / count:.4f}")
    else:
        print("  No valid diagnosis_metrics found.")

    # 2. LLM scores
    total_diag = 0.0
    total_overall = 0.0
    llm_count = 0
    for result in results:
        ev = result.get("evaluation", {})
        if isinstance(ev, dict):
            ds = ev.get("diagnosis_score")
            os_ = ev.get("overall_score")
            if isinstance(ds, (int, float)):
                total_diag += ds
                llm_count += 1
            if isinstance(os_, (int, float)):
                total_overall += os_

    print("\n========== Judge Model Scores ==========")
    if llm_count > 0:
        print(f"  Diagnosis Score: {total_diag / llm_count:.2f} / 5")
        print(f"  Overall Score  : {total_overall / llm_count:.2f} / 5")
    else:
        print("  No valid judge scores found.")


if __name__ == "__main__":
    main()
