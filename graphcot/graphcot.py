"""
graphcot: Advanced QA Generation Module
-------------------------------------------

This module provides optimized algorithms for long-tail entity partitioning
and Chain-of-Thought (CoT) enhanced QA generation.

It is designed to be standalone, with minimal external dependencies.
"""

import abc
import random
import re
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union, Optional

import numpy as np

# --- Core Data Structures & Interfaces ---

@dataclass
class Community:
    """Represents a partitioned subgraph community."""
    id: Union[int, str]
    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseGraphStorage(abc.ABC):
    """Abstract interface for graph data access."""
    
    @abc.abstractmethod
    def get_all_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return list of (node_id, node_data)."""
        pass

    @abc.abstractmethod
    def get_all_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return list of (u, v, edge_data)."""
        pass


class BaseLLMClient(abc.ABC):
    """Abstract interface for LLM interaction."""
    
    @abc.abstractmethod
    async def generate_answer(self, prompt: str, **kwargs: Any) -> str:
        """Generate text response from LLM.
        
        Args:
            prompt: The input prompt text.
            **kwargs: Additional generation parameters (e.g., temperature).
            
        Returns:
            The generated text response.
        """
        pass


def compute_content_hash(content: str) -> str:
    """Compute MD5 hash for content ID generation."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# --- Partitioner Implementation ---

class LongTailPartitioner:
    """
    Partitioner designed to boost coverage of long-tail (sparse) entities.
    
    Features:
    1. Inverse Degree Sampling: Prioritizes low-degree entities.
    2. Adaptive Expansion: Expands search radius for sparse entities.
    """

    def partition(
        self,
        g: BaseGraphStorage,
        max_size: int = 50,
        min_size: int = 3,
        tail_ratio: float = 0.5,
        expansion_depth: int = 2,
        **kwargs: Any,
    ) -> List[Community]:
        """
        Partition graph into communities with focus on long-tail entities.
        
        Args:
            g: Graph storage instance.
            max_size: Maximum nodes per community.
            min_size: Minimum nodes per community.
            tail_ratio: Ratio of communities centered on tail entities.
            expansion_depth: Max depth for BFS expansion on tail nodes.
            
        Returns:
            List of Community objects.
        """
        all_nodes = g.get_all_nodes() or []
        all_edges = g.get_all_edges() or []
        
        if not all_nodes:
            return []

        # Build graph index
        adj, edge_index, degrees = self._build_index(all_nodes, all_edges)
        
        if not degrees:
            return []

        # Classify and sample anchors
        tail_anchors, head_anchors = self._sample_anchors(
            degrees, tail_ratio, kwargs.get("max_communities", 100)
        )

        # Expand communities from anchors
        return self._expand_communities(
            tail_anchors, head_anchors, adj, edge_index,
            degrees, max_size, min_size, expansion_depth
        )

    @staticmethod
    def _build_index(
        all_nodes: List[Tuple[str, Dict[str, Any]]],
        all_edges: List[Tuple[str, str, Dict[str, Any]]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, str, Any]]], Dict[str, int]]:
        """Build adjacency list, edge index, and degree map.
        
        Returns:
            (adj, edge_index, degrees) where edge_index maps node_id -> list of
            (u, v, data) edges touching that node, enabling O(1) lookup per node.
        """
        adj: Dict[str, List[str]] = defaultdict(list)
        edge_index: Dict[str, List[Tuple[str, str, Any]]] = defaultdict(list)

        for u, v, edata in all_edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_index[u].append((u, v, edata))
            edge_index[v].append((u, v, edata))

        degrees = {nid: len(adj[nid]) for nid, _ in all_nodes}
        return dict(adj), dict(edge_index), degrees

    @staticmethod
    def _sample_anchors(
        degrees: Dict[str, int],
        tail_ratio: float,
        total_budget: int,
    ) -> Tuple[List[str], List[str]]:
        """Sample tail and head anchor nodes using inverse degree weighting."""
        sorted_degrees = sorted(degrees.values())
        median_degree = sorted_degrees[len(sorted_degrees) // 2]

        tail_nodes = [nid for nid, deg in degrees.items() if 0 < deg <= median_degree]
        head_nodes = [nid for nid, deg in degrees.items() if deg > median_degree]

        num_tail = int(total_budget * tail_ratio)
        num_head = total_budget - num_tail

        # Weighted sampling for tail nodes (lower degree -> higher probability)
        tail_anchors: List[str] = []
        if tail_nodes:
            weights = [1.0 / (degrees[n] + 1e-6) for n in tail_nodes]
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                sampled_indices = np.random.choice(
                    len(tail_nodes),
                    size=min(num_tail, len(tail_nodes)),
                    replace=False,
                    p=probs,
                )
                tail_anchors = [tail_nodes[i] for i in sampled_indices]
            else:
                tail_anchors = random.sample(tail_nodes, k=min(num_tail, len(tail_nodes)))

        # Random sampling for head nodes
        head_anchors: List[str] = []
        if head_nodes:
            head_anchors = random.sample(head_nodes, k=min(num_head, len(head_nodes)))

        return tail_anchors, head_anchors

    @staticmethod
    def _bfs_expand(
        anchor: str,
        adj: Dict[str, List[str]],
        max_depth: int,
        max_size: int,
    ) -> Set[str]:
        """BFS expansion from anchor up to max_depth / max_size."""
        visited = {anchor}
        queue = deque([(anchor, 0)])
        while queue and len(visited) < max_size:
            curr, depth = queue.popleft()
            if depth >= max_depth:
                continue
            neighbors = list(adj.get(curr, []))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    if len(visited) >= max_size:
                        break
        return visited

    @staticmethod
    def _collect_edges(
        nodes: Set[str],
        edge_index: Dict[str, List[Tuple[str, str, Any]]],
    ) -> List[Tuple[str, str, Any]]:
        """Collect unique internal edges for a node set using the edge index.
        
        Uses edge_index for O(sum(deg(v))) lookup instead of O(E) full scan.
        """
        seen: Set[Tuple[str, str]] = set()
        result: List[Tuple[str, str, Any]] = []
        for nid in nodes:
            for u, v, data in edge_index.get(nid, []):
                if u in nodes and v in nodes:
                    key = (min(u, v), max(u, v))
                    if key not in seen:
                        seen.add(key)
                        result.append((u, v, data))
        return result

    def _expand_communities(
        self,
        tail_anchors: List[str],
        head_anchors: List[str],
        adj: Dict[str, List[str]],
        edge_index: Dict[str, List[Tuple[str, str, Any]]],
        degrees: Dict[str, int],
        max_size: int,
        min_size: int,
        expansion_depth: int,
    ) -> List[Community]:
        """Expand anchor nodes into communities via BFS."""
        communities: List[Community] = []
        comm_id = 0

        # Tail anchors get multi-hop expansion
        for anchor in tail_anchors:
            nodes_in_comm = self._bfs_expand(anchor, adj, expansion_depth, max_size)
            if len(nodes_in_comm) >= min_size:
                communities.append(Community(
                    id=comm_id,
                    nodes=list(nodes_in_comm),
                    edges=self._collect_edges(nodes_in_comm, edge_index),
                    metadata={"type": "long_tail", "anchor": anchor, "degree": degrees[anchor]},
                ))
                comm_id += 1

        # Head anchors get standard 1-hop expansion
        for anchor in head_anchors:
            nodes_in_comm = self._bfs_expand(anchor, adj, max_depth=1, max_size=max_size)
            if len(nodes_in_comm) >= min_size:
                communities.append(Community(
                    id=comm_id,
                    nodes=list(nodes_in_comm),
                    edges=self._collect_edges(nodes_in_comm, edge_index),
                    metadata={"type": "head", "anchor": anchor, "degree": degrees[anchor]},
                ))
                comm_id += 1

        return communities


# --- Generator Implementation ---


COT_PROMPT_TEMPLATE = """---角色---
你是一位严谨、保守、可审计的呼吸医学知识问答数据生成专家，同时具备临床教学能力。
你的目标不是“尽量多生成”，而是“只输出证据充分、医学安全、可用于SFT的高质量样本”。
你可在内部利用图谱证据推理，但最终 <thought> 必须是自然中文医学讲解，不能暴露图谱检索过程。

---任务目标---
基于【医学图谱上下文】（节点与关系边），生成最多 {num_qa} 个高质量中文医学问答对（QA）。
先在内部构造候选样本并完成审计，再只输出通过样本。允许少于 {num_qa}，宁缺毋滥。

优先覆盖三类样本：
1. 常见 QA（基础认知）：单跳/双跳，答案直接、明确、尽量唯一。
2. 长尾 QA（复杂推理）：至少融合 3 个图谱实体，且具备可审计的多跳判别结构。
3. 判别 QA（选择/排除）：必须给出一个“错误但看似合理”的候选答案并可被证据明确排除。

样本分布目标（用于训练决策能力）：
- 判别 QA 至少占 30%
- 长尾 QA 至少占 30%
- 其余可为常见 QA
若当前图谱证据无法满足占比，可减少总量，也不得放松质量门槛。

---图谱上下文---
{graph_context}

---质量优先级---
1. 医学语义正确性 > 图谱忠实度 > 医学安全性 > 题目数量
2. 证据不足时，必须降级或放弃，不得臆测补全
3. 可审计性与可回指性必须优先于语言华丽度

---硬性约束---
【A. 图谱忠实】
1. 只能使用图谱上下文中明确出现的实体与关系。
2. 不得创造图谱外实体、别名、关系、机制、治疗方案、检查结论、诊断结论。
3. 必须保持关系方向；严禁把“相关”写成“因果”、把“预防”写成“治疗”、把“不良反应”写成“适应症”、把“慎用”写成“禁用”。
4. 不得写入图谱未给出的剂量、疗程、首选级别、指南等级、证据等级。
5. 开放世界约束（OWA）：图谱未出现仅表示“未知”，不得据此推出“不存在/不适用/应排除”。
6. 若实体间无显式关系，且无法通过显式中间节点形成路径，不得自行补全。
7. 若证据不足以支持结论，该 QA 必须丢弃。

【B. 医学安全】
1. 若图谱存在禁忌、慎用、严重不良反应、高危人群限制，必须优先用于排除错误结论。
2. 特殊人群结论（儿童、孕妇、老年人、免疫缺陷、肝肾异常等）仅在图谱明确给出时可写入。
3. 不得把“可能”升级为“确定”，不得把“可考虑”升级为“首选/更优”。
4. 若图谱无显式替代/排序/优选依据，不得使用“更合适/更安全/优先/首选/最佳”等排序性结论。
5. 严禁输出可能造成严重误导的建议（违背禁忌、混淆不良反应与疗效、证据不足却给确定性诊断或用药建议）。

【C. 推理与判别性】
1. 常见题至少 1 条显式证据链；长尾/判别题至少 2 条显式证据链，且至少 3 个实体参与。
2. 判别题为强约束：必须同时满足“支持正确答案 + 排除错误候选答案（distractor）”。
3. 长尾题优先构造 distractor；若题型属于比较/排除/选择型，则必须提供可排除依据。
4. 常见题不强制构造 distractor，不得为了凑判别而改写成伪选择题。
5. 若题目属于比较/排除/选择型，但 evidence 不能排除合理替代结论，该样本不通过。
6. 任何排除/比较/替代判断都必须由显式证据支持，禁止基于“缺失信息”构造否定结论。
7. answer 的核心断言必须可由 evidence 直接支持；允许受控医学表达归纳，不要求逐字贴三元组。
8. 禁止信息空洞型答案（如“存在风险”“需谨慎”）；答案至少包含一个具体医学对象，并体现关键关系语义。
9. 若问题不依赖图谱关键关系、仅凭常识即可回答，该 QA 必须判为无效并丢弃。
10. 禁止循环推理：不得使用“已知A再证明A”的逻辑空转，结论必须引入新增可判别信息。
11. 不同 QA 之间避免重复考察同一证据链。

【D. 受控语义解释（仅限在证据边界内）】
1. 允许在不新增实体、不新增因果关系的前提下做有限医学语义解释（如表现的同层语义归纳、在既有疾病/用药背景下的保守归因）。
2. 语义解释只能用于提高可读性，不能扩展为新治疗结论、新诊断结论或机制细节。
3. 若语义解释导致 evidence 无法逐项回指，必须回退并丢弃该样本。

---生成流程（必须按顺序）---
步骤1：证据链选择
- 常见题：选 1 条以上显式证据链。
- 长尾/判别题：优先组合多条证据形成“支持+排除/比较”的判别结构。

步骤2：充分性与判别性检查
- 充分性：能否支持明确、安全、唯一或高可判别答案。
- 判别性：是否能排除至少一个语义上合理的替代结论。
- 不满足则降级或丢弃，不得硬凑。

步骤2.5：错误候选构造（分级）
- 判别题：必须构造一个错误但合理的 distractor，并可被 evidence 明确排除。
- 长尾题：优先构造 distractor；若题型是比较/排除/选择型则必须构造。
- 常见题：不强制构造 distractor，不得为了凑 distractor 牺牲事实召回。
- 若应构造而无法构造可被证据排除的 distractor，则该样本丢弃。

步骤3：先定结论边界，再写问题
- 先确定图谱最多支持到哪一步，再围绕该边界设计问题。
- 禁止关系类型升级（相关→因果、可用→推荐、提示→诊断）。

步骤4：生成 <thought>
- 写成连续自然的中文医学推理（建议 80-220 字）。
- 允许提及排除/比较依据，但不得出现图谱检索痕迹或三元组罗列。
- 禁止出现“根据知识图谱/证据链如下/Step1-2-3/实体A-关系-实体B”等表达。

步骤5：生成 <answer>
- 短、准、可回指到 evidence。
- 不得引入 evidence 未覆盖的医学判断。

步骤6：反事实一致性审计（强制）
- 构造至少一个基于已出现实体/关系的替代候选结论（非“缺失即否定”）。
- 检查 evidence 是否仅支持当前答案并能排除替代结论。
- 若无法区分，则该 QA 丢弃。

---输出格式要求---
1. 仅输出 XML，不得输出任何额外说明文字。
2. 所有内容必须为中文。
3. 每个 qa 必须包含：type、question、thought、answer、evidence、quality。
4. distractor：判别题必填，长尾题可选，常见题可省略。
5. evidence 使用三元组格式：实体A|关系|实体B。
6. XML 安全：若文本包含 &, <, >，必须转义为 &amp;、&lt;、&gt;。
7. 若无合格样本，输出：
<qas></qas>

请严格使用以下结构：
<qas>
<qa>
<type>常见|长尾|判别</type>
<question>...</question>
<thought>...</thought>
<answer>...</answer>
<distractor>...</distractor>
<evidence>
<triple>实体A|关系|实体B</triple>
<triple>实体C|关系|实体D</triple>
</evidence>
<quality>
<faithfulness>high</faithfulness>
<safety>pass</safety>
<uniqueness>pass</uniqueness>
</quality>
</qa>
</qas>

---输出前最终自检---
在输出每个 QA 前，逐项确认：
- 仅使用图谱中存在的实体与关系
- 关系方向与关系类型未被混淆或升级
- 无图谱外事实、无不受控常识补桥（受控语义解释除外）
- answer 的核心断言可被 evidence 直接支持或受控归纳支持
- 判别题的 distractor 为“看似合理但可被证据排除”的错误结论
- 证据不足样本已丢弃，未输出
- 问题自然、答案明确、医学安全
- 若证据不足或不可判别，已丢弃该样本
"""

class CoTGenerator:
    """
    Generates QA pairs with explicit Chain-of-Thought (CoT) reasoning steps.
    """

    def __init__(self, llm_client: BaseLLMClient, num_of_questions: int = 2):
        self.llm_client = llm_client
        self.num_of_questions = num_of_questions

    def build_prompt(
        self,
        nodes: List[Tuple[str, Dict[str, Any]]], 
        edges: List[Tuple[str, str, Dict[str, Any]]],
        num_qa: int = 2,
        community_type: str = "mixed"
    ) -> str:
        """Build prompt with graph context."""
        context_lines = []
        node_map = {}
        
        # Format nodes
        for nid, data in nodes:
            name = data.get("name", nid)
            etype = data.get("entity_type", "实体")
            node_map[nid] = name
            context_lines.append(f"- [{etype}] {name}")
            
        # Format edges
        for u, v, data in edges:
            u_name = node_map.get(u, u)
            v_name = node_map.get(v, v)
            rel = data.get("relation_type") or data.get("relation_name", "相关")
            context_lines.append(f"{u_name} --[{rel}]--> {v_name}")
            
        graph_context = "\n".join(context_lines[:180])  # Keep more evidence for better QA quality
        
        # Add hint about community type if needed
        hint = ""
        if community_type == "head":
            hint = "\n【难度侧重提示】：当前提供的图谱上下文主要包含基础、核心的医学实体。请优先生成【常见 QA】（基础认知题）。\n"
        elif community_type == "long_tail":
            hint = "\n【难度侧重提示】：当前提供的图谱上下文包含罕见或复杂的长尾医学实体。请优先生成【长尾 QA】（结合并发症/禁忌症的多跳复杂推理题）。\n"

        return COT_PROMPT_TEMPLATE.format(
            num_qa=num_qa,
            graph_context=hint + graph_context
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse QA pairs with thoughts."""
        result = {}
        qa_blocks = re.findall(r"<qa>(.*?)</qa>", response, re.DOTALL)

        def _extract_tag(block: str, tag: str) -> str:
            match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", block, re.DOTALL)
            return match.group(1).strip() if match else ""

        for block in qa_blocks:
            qa_type = _extract_tag(block, "type") or "未分类"
            question = _extract_tag(block, "question")
            thought = _extract_tag(block, "thought")
            answer = _extract_tag(block, "answer")
            distractor = _extract_tag(block, "distractor")
            evidence_block = _extract_tag(block, "evidence")
            evidence = [e.strip() for e in re.findall(r"<triple>(.*?)</triple>", evidence_block)] if evidence_block else []

            if question and answer:
                qa_id = compute_content_hash(question)
                qa_item: Dict[str, Any] = {
                    "type": qa_type,
                    "question": question,
                    "thought": thought,
                    "answer": answer,
                    "evidence": evidence,
                }
                if distractor:
                    qa_item["distractor"] = distractor
                result[qa_id] = qa_item

        # Fallback: tolerate malformed XML blocks without full <qa> wrapper
        if not result:
            qa_blocks_fallback_basic = re.findall(
                r"<qa>\s*<question>(.*?)</question>\s*<thought>(.*?)</thought>\s*<answer>(.*?)</answer>.*?</qa>",
                response,
                re.DOTALL,
            )
            for question, thought, answer in qa_blocks_fallback_basic:
                question = question.strip()
                answer = answer.strip()
                if question and answer:
                    qa_id = compute_content_hash(question)
                    result[qa_id] = {
                        "type": "未分类",
                        "question": question,
                        "thought": thought.strip(),
                        "answer": answer,
                        "evidence": [],
                    }
        return result

    async def generate_from_data(
        self,
        nodes: List[Tuple[str, Dict[str, Any]]],
        edges: List[Tuple[str, str, Dict[str, Any]]],
        community_type: str = "mixed"
    ) -> Dict[str, Any]:
        """Generate QA from explicit node/edge data."""
        prompt = self.build_prompt(nodes, edges, num_qa=self.num_of_questions, community_type=community_type)
        response = await self.llm_client.generate_answer(prompt)
        return self.parse_response(response)
