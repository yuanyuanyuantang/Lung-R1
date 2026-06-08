"""
Medical guideline OCR document data cleaning script

Description:
    Use LLM to clean OCR-converted medical guideline PDF documents, removing
    irrelevant content (authors, references, citation markers, headers/footers,
    etc.) while preserving core medical information (diseases, symptoms, treatment
    plans, drug dosages, recommendation levels, etc.) for subsequent knowledge
    graph construction.

Input path:
    ./准备数据/1.原始指南数据/<source_hospital>提供的指南数据/  (OCR-converted raw txt files)

Output path:
    ./准备数据/2.<source_hospital>提供的指南数据_cleaned/       (Cleaned txt files)
    ./准备数据/2.<source_hospital>提供的指南数据_cleaned/audit_logs/  (Audit logs)

Usage:
    python data_cleaning.py                    # Batch clean all files
    python data_cleaning.py --preview file.txt # Preview cleaning result for a single file
    python data_cleaning.py --resume           # Resume from checkpoint (skip completed files)
"""

import os, json, asyncio, re, logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

# ==================== Configuration ====================

@dataclass
class Config:
    input_dir: str = "./准备数据/1.原始指南数据/<source_hospital>提供的指南数据"
    output_dir: str = "./准备数据/2.<source_hospital>提供的指南数据_cleaned"
    api_key: str = "***"  # Replace with your own API Key
    api_base: str = "https://your-api-endpoint/v1"
    model: str = "your-model-name"
    chunk_size: int = 3500
    max_retries: int = 3

CFG = Config()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

PROMPT_CLEAN = """你是医学文档清洗专家。请对以下OCR文本进行清洗，用于构建医学知识图谱。

【核心原则 - 极其重要】
1. 只做删除，绝对不要改写、补全、润色任何医学内容
2. 保留原文的所有医学表述，包括不通顺的句子
3. 数字、剂量、单位、百分比必须原样保留
4. 推荐等级（如A级、B级、I类、II类、AI、AII、BI、BII等各种形式）必须原样保留
5. 否定表述（不推荐、禁忌、避免、慎用等）必须原样保留

【需要删除的内容】
- 作者姓名、作者单位、通讯作者、邮箱
- 参考文献列表（References、参考文献）
- 引用标记：[1]、[1,2]、[1-5]、(1)、(1,2) 等
- 网站链接、DOI、URL
- 图注（Figure 1、图1、Fig.）、表注说明
- 版权声明、出版信息、期刊名称
- 页眉页脚、页码
- 缩写说明段落（Abbreviations: ...）
- 致谢、利益冲突、资金来源声明
- LaTeX残留符号（$、\\mathrm、\\等）
- 明显的OCR乱码（无意义字符序列）

【必须保留的内容 - 即使格式混乱也要保留】
- 所有疾病名称、症状描述
- 所有诊断标准、检查方法
- 所有治疗方案、药物名称、剂量、疗程
- 所有推荐等级、证据级别
- 所有禁忌症、不良反应、注意事项
- 表格中的医学数据内容

【输出要求】
- 直接输出清洗后的文本
- 不要添加任何解释、总结、标记
- 不要改写任何保留的内容

原始文本：
{text}"""

PROMPT_VERIFY = """对比医学文档清洗前后，统计关键信息保留情况。

【原文片段】
{original}

【清洗后片段】
{cleaned}

请统计以下4类信息在原文和清洗后的出现次数，输出JSON：
- 剂量信息：药物剂量如 mg、ml、g/kg 等
- 推荐等级：如 A级、I类、强推荐、1A 等
- 否定表述：不推荐、禁忌、避免、慎用等
- 数值指标：检验阈值、诊断标准数值

严格按此格式输出，不要有其他内容：
{{"剂量信息":{{"原文":0,"清洗后":0}},"推荐等级":{{"原文":0,"清洗后":0}},"否定表述":{{"原文":0,"清洗后":0}},"数值指标":{{"原文":0,"清洗后":0}},"问题":[]}}"""

# ==================== Core functions ====================

def preprocess(text: str) -> str:
    """Preprocess OCR text, normalize line breaks"""
    lines, result = text.split('\n'), []

    for line in lines:
        s = line.strip()
        if not s:
            result.append('')
            continue

        # Structured lines (lists/headings/short lines) are not merged
        is_struct = (
            s[0] in '-•●○·*－' or
            re.match(r'^\d{1,3}[.、)）]', s) or  # Support multi-digit numbering like 10. 12.
            re.match(r'^\([0-9a-zA-Z]+\)', s) or  # (1) (a) etc.
            s.startswith(('#', 'Table', '表')) or
            s.isupper() or len(s) < 20
        )

        if is_struct or not result or not result[-1]:
            result.append(s)
        elif result[-1][-1] not in '。．.!?！？:：；;':
            result[-1] += ' ' + s
        else:
            result.append(s)

    # Segment by empty lines
    paras, current = [], []
    for line in result:
        if line:
            current.append(line)
        elif current:
            paras.append('\n'.join(current))
            current = []
    if current:
        paras.append('\n'.join(current))

    return '\n\n'.join(paras)


def split_chunks(text: str, size: int = 3500, overlap: int = 150) -> list[str]:
    """Smart chunking, split at sentence boundaries with overlap"""
    text = preprocess(text)
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, current = [], ""

    for para in paras:
        if len(para) > size:  # Overly long paragraphs split by sentence
            if current:
                chunks.append(current.strip())
                current = ""

            # Split by sentence (avoid breaking abbreviations and decimals)
            # Only split at "punctuation+space+uppercase" or "Chinese punctuation"
            sents = re.split(r'([。！？]|(?<=[^A-Z0-9])[.!?](?=\s+[A-Z]))', para)
            temp, added = "", False
            i = 0
            while i < len(sents):
                sent = sents[i]
                # Reattach punctuation to the previous sentence
                if i + 1 < len(sents) and len(sents[i+1]) <= 1:
                    sent += sents[i+1]
                    i += 1

                if len(temp) + len(sent) < size:
                    temp += sent
                else:
                    if temp:
                        chunks.append(temp.strip())
                        added = True
                        # Overlap: keep trailing portion for next chunk
                        if len(temp) > overlap:
                            temp = temp[-overlap:] + sent
                        else:
                            temp = sent
                    else:
                        temp = sent
                i += 1

            if temp:
                chunks.append(temp.strip())
                added = True

            # Fallback: if sentence splitting produced nothing, hard-split by character
            if not added:
                for i in range(0, len(para), size - overlap):
                    chunks.append(para[i:i + size])

        elif len(current) + len(para) + 2 < size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"

    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


async def call_llm(text: str, client, idx: int = 0) -> tuple[str, bool]:
    """Call LLM for cleaning, with retry"""
    for attempt in range(CFG.max_retries):
        try:
            resp = await client.chat.completions.create(
                model=CFG.model,
                messages=[{"role": "user", "content": PROMPT_CLEAN.format(text=text)}],
                temperature=0.0,
                max_tokens=8000,
            )
            return resp.choices[0].message.content.strip(), True
        except Exception as e:
            log.warning(f"Chunk {idx} attempt {attempt+1} failed: {e}")
            if attempt < CFG.max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
    return text, False


async def verify_cleaning(original: str, cleaned: str, client) -> dict:
    """Use LLM to verify cleaning quality, check if key information was accidentally deleted"""
    # Truncate content to avoid excessive length
    orig_sample = original[:3500] if len(original) > 3500 else original
    clean_sample = cleaned[:3500] if len(cleaned) > 3500 else cleaned

    try:
        resp = await client.chat.completions.create(
            model=CFG.model,
            messages=[{"role": "user", "content": PROMPT_VERIFY.format(original=orig_sample, cleaned=clean_sample)}],
            temperature=0.0,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content.strip()

        # Handle possible markdown code blocks
        if "```" in content:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)

        # Try to extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            content = json_match.group(0)

        return json.loads(content)
    except json.JSONDecodeError as e:
        log.warning(f"Verification result JSON parsing failed: {e}")
        # Return raw content for debugging
        return {"error": "JSON parsing failed", "raw": content[:500] if 'content' in dir() else "No content"}
    except Exception as e:
        log.warning(f"Verification of cleaning quality failed: {e}")
        return {"error": str(e)}


async def clean_doc(text: str) -> tuple[str, dict]:
    """Clean an entire document"""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=CFG.api_key, base_url=CFG.api_base)

    try:
        chunks = split_chunks(text, CFG.chunk_size)
        audit = {"total": len(chunks), "failed": [], "stats": [], "verification": None}

        sem = asyncio.Semaphore(3)
        async def process(chunk, idx):
            async with sem:
                log.info(f"  Cleaning chunk {idx+1}/{len(chunks)}")
                return (*await call_llm(chunk, client, idx), idx, len(chunk))

        results = await asyncio.gather(*[process(c, i) for i, c in enumerate(chunks)])

        cleaned = []
        for text_result, ok, idx, orig_len in results:
            cleaned.append(text_result)
            audit["stats"].append({"idx": idx, "orig": orig_len, "clean": len(text_result), "ok": ok})
            if not ok:
                audit["failed"].append(idx)

        # Merge and remove duplicate paragraphs caused by overlap
        merged = "\n\n".join(cleaned)
        merged = remove_duplicate_paragraphs(merged)

        # LLM verification of cleaning quality
        log.info("  Verifying cleaning quality...")
        audit["verification"] = await verify_cleaning(text, merged, client)

        return merged, audit
    finally:
        await client.close()


def remove_duplicate_paragraphs(text: str, min_len: int = 50) -> str:
    """Remove duplicate paragraphs (which may be caused by overlap)"""
    paras = text.split('\n\n')
    seen, result = set(), []

    for p in paras:
        p = p.strip()
        if not p:
            continue
        # Short paragraphs are not deduplicated (may be legitimate repeats like headings)
        if len(p) < min_len:
            result.append(p)
        elif p not in seen:
            seen.add(p)
            result.append(p)

    return '\n\n'.join(result)

# ==================== Audit logs ====================

def save_audit(out_dir: Path, name: str, orig: str, clean: str, audit: dict):
    """Save audit logs and backups"""
    stem = Path(name).stem

    # Backup original
    (out_dir / "original_backup").mkdir(parents=True, exist_ok=True)
    (out_dir / "original_backup" / name).write_text(orig, encoding='utf-8')

    # Audit logs
    log_dir = out_dir / "audit_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "file": name, "time": datetime.now().isoformat(),
        "orig_len": len(orig), "clean_len": len(clean),
        "reduction": round((1 - len(clean)/len(orig)) * 100, 2) if orig else 0,
        "chunks": audit
    }
    (log_dir / f"{stem}_audit.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    # Diff summary
    diff = [f"File: {name}", f"Chars: {len(orig)} -> {len(clean)} (reduced {meta['reduction']}%)", "=" * 50, ""]

    # Check deleted content
    diff.append("[Deletion Detection]")
    keywords_to_check = [
        ("References", "参考文献"),
        ("Conflict", "利益冲突"),
        ("Funding", "资金"),
        ("Acknowledgment", "致谢"),
        ("Author", "作者"),
    ]
    for en, zh in keywords_to_check:
        was = en.lower() in orig.lower() or zh in orig
        now = en.lower() in clean.lower() or zh in clean
        if was and not now:
            diff.append(f"  ✓ Deleted: {en}/{zh}")
        elif was:
            diff.append(f"  ⚠ Still present: {en}/{zh}")

    # LLM verification results
    diff.append("\n[Retention Check - LLM Verification]")
    verification = audit.get("verification", {})
    if "error" in verification:
        diff.append(f"  ⚠ Verification failed: {verification['error']}")
        if "raw" in verification:
            diff.append(f"  Raw output: {verification['raw'][:200]}...")
    else:
        for category in ["剂量信息", "推荐等级", "否定表述", "数值指标"]:
            if category in verification:
                v = verification[category]
                orig_cnt = v.get("原文", 0)
                clean_cnt = v.get("清洗后", 0)
                if orig_cnt > 0:
                    ratio = clean_cnt / orig_cnt * 100
                    status = "✓" if ratio >= 70 else "⚠"
                    diff.append(f"  {status} {category}: {orig_cnt}->{clean_cnt} ({ratio:.0f}%)")
                else:
                    diff.append(f"  - {category}: No such info in original")

        # Show problems
        problems = verification.get("问题", [])
        if problems:
            diff.append("\n[Potential Issues]")
            for p in problems:
                diff.append(f"  ⚠ {p}")

    (log_dir / f"{stem}_diff.txt").write_text('\n'.join(diff), encoding='utf-8')

# ==================== Progress management ====================

def load_progress(out_dir: Path) -> set:
    f = out_dir / ".progress.json"
    if not f.exists():
        return set()
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        return set(data.get("done", []))
    except (json.JSONDecodeError, KeyError):
        log.warning("Progress file corrupted, restarting from scratch")
        return set()

def save_progress(out_dir: Path, done: set):
    (out_dir / ".progress.json").write_text(json.dumps({"done": list(done)}), encoding='utf-8')

# ==================== Main flow ====================

def process_file(inp: Path, out: Path) -> dict:
    """Process a single file"""
    content = inp.read_text(encoding='utf-8')
    cleaned, audit = asyncio.run(clean_doc(content))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cleaned, encoding='utf-8')
    save_audit(out.parent, inp.name, content, cleaned, audit)

    return {
        "file": inp.name,
        "orig": len(content),
        "clean": len(cleaned),
        "failed": len(audit["failed"])
    }


def process_dir(inp_dir: str, out_dir: str, resume: bool = False):
    """Batch process directory"""
    inp, out = Path(inp_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = list(inp.glob("*.txt"))
    log.info(f"Found {len(files)} files")

    done = load_progress(out) if resume else set()
    if done:
        log.info(f"Skipping {len(done)} already completed files")

    results = []
    for i, f in enumerate(files, 1):
        if f.name in done:
            continue

        log.info(f"[{i}/{len(files)}] {f.name}")
        try:
            r = process_file(f, out / f.name)
            results.append(r)
            log.info(f"  {r['orig']} -> {r['clean']} chars" + (f" [Warning: {r['failed']} chunks failed]" if r['failed'] else ""))
            done.add(f.name)
            save_progress(out, done)
        except Exception as e:
            log.error(f"  Failed: {e}")

    # Summary
    if results:
        total_o = sum(r['orig'] for r in results)
        total_c = sum(r['clean'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        log.info(f"Completed {len(results)} files, {total_o} -> {total_c} chars (reduced {(1-total_c/total_o)*100:.1f}%)")
        if total_failed > 0:
            log.warning(f"⚠ Total {total_failed} chunks failed to clean, original content preserved, recommend checking")


def preview(path: str):
    """Preview cleaning result"""
    content = Path(path).read_text(encoding='utf-8')[:3000]
    cleaned, audit = asyncio.run(clean_doc(content))

    print("=" * 50, "\n[Original]\n", "=" * 50)
    print(content)
    print("\n" + "=" * 50, "\n[Cleaned]\n", "=" * 50)
    print(cleaned)
    print(f"\n{len(content)} -> {len(cleaned)} chars, {audit['total']} chunks, {len(audit['failed'])} failed")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=CFG.input_dir)
    p.add_argument("--output", default=CFG.output_dir)
    p.add_argument("--preview", type=str)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.preview:
        preview(args.preview)
    else:
        process_dir(args.input, args.output, args.resume)
