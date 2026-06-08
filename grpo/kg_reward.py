"""
KG-Guided Reward Function for GRPO.

Three components:
  R_dx:    Diagnosis correctness reward (outcome-level)
  R_graph: Graph faithfulness reward (process-level)
  R_path:  Relation/path consistency reward (process-level)

Final: R = λ₁·R_dx + λ₂·R_graph + λ₃·R_path
"""

import re
import pandas as pd
import networkx as nx
from typing import List, Dict, Set, Tuple, Optional


class KGRewardFunction:
    """KG-guided reward function for pulmonary diagnosis GRPO training."""

    def __init__(
        self,
        nodes_path: str,
        edges_path: str,
        lambda_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        self.lambda_dx, self.lambda_graph, self.lambda_path = lambda_weights

        # ── Load KG ──
        print("[KG Reward] Loading knowledge graph...")
        self._load_kg(nodes_path, edges_path)
        print(f"[KG Reward] Loaded {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges, "
              f"{len(self.valid_relation_types)} relation types")

        # ── Build entity name index for fast lookup ──
        self._build_entity_index()

    # ═══════════════════════════════════════════════════════════════
    # KG Loading
    # ═══════════════════════════════════════════════════════════════

    def _load_kg(self, nodes_path: str, edges_path: str):
        """Load knowledge graph from CSV files into NetworkX."""
        # Load nodes
        nodes_df = pd.read_csv(nodes_path)
        self.G = nx.DiGraph()

        for _, row in nodes_df.iterrows():
            node_id = row[":ID"]
            name = str(row["name"]) if pd.notna(row["name"]) else ""
            etype = str(row["type"]) if pd.notna(row["type"]) else ""
            desc = str(row["description"]) if pd.notna(row["description"]) else ""
            self.G.add_node(node_id, name=name, type=etype, description=desc)

        # Load edges
        edges_df = pd.read_csv(edges_path)
        self.valid_relation_types = set()

        for _, row in edges_df.iterrows():
            start_id = row[":START_ID"]
            end_id = row[":END_ID"]
            rel_type = str(row[":TYPE"]) if pd.notna(row[":TYPE"]) else ""
            rel_name = str(row["relation_name"]) if pd.notna(row["relation_name"]) else ""
            weight = float(row["weight:float"]) if pd.notna(row["weight:float"]) else 1.0

            if start_id in self.G and end_id in self.G:
                self.G.add_edge(start_id, end_id,
                                relation_type=rel_type,
                                relation_name=rel_name,
                                weight=weight)
                if rel_type:
                    self.valid_relation_types.add(rel_type)
                if rel_name:
                    self.valid_relation_types.add(rel_name)

    def _build_entity_index(self):
        """Build name-based index for fast entity lookup.

        Maps entity names to node IDs. Handles duplicates by storing all matches.
        Also pre-computes a k-hop neighbor cache for fast KG path checks.
        """
        self.entity_names: Dict[str, List[str]] = {}  # name -> [node_id, ...]
        self.all_entity_names: List[str] = []  # sorted by length desc for longest-match

        for node_id, data in self.G.nodes(data=True):
            name = data.get("name", "")
            if name and name != "nan":
                if name not in self.entity_names:
                    self.entity_names[name] = []
                self.entity_names[name].append(node_id)
                self.all_entity_names.append(name)

        # Sort by length descending for longest-match-first
        seen = set()
        unique_names = []
        for name in sorted(self.all_entity_names, key=len, reverse=True):
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        self.all_entity_names = unique_names

        # Pre-compute 1-hop neighbors for fast edge/path checks
        print("[KG Reward] Building 1-hop neighbor cache...")
        self._onehop = {}
        for node_id in self.G.nodes():
            neighbors = set(self.G.successors(node_id)) | set(self.G.predecessors(node_id))
            self._onehop[node_id] = neighbors
        print("[KG Reward] Neighbor cache ready")

    def _has_kg_connection(self, n1: str, n2: str, max_hops: int = 2) -> bool:
        """Fast check: are n1 and n2 connected within max_hops in the KG?

        Uses pre-computed 1-hop neighbors for O(degree) lookup instead of
        full shortest-path search.
        """
        if n1 not in self._onehop or n2 not in self._onehop:
            return False
        # Direct edge
        if n2 in self._onehop[n1]:
            return True
        if max_hops < 2:
            return False
        # 2-hop: check if they share a neighbor
        if self._onehop[n1] & self._onehop[n2]:
            return True
        return False

    # ═══════════════════════════════════════════════════════════════
    # Main Reward Interface
    # ═══════════════════════════════════════════════════════════════

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truth: List[str] = None,
        **kwargs,
    ) -> List[float]:
        """Compute KG-guided rewards for a batch of completions.

        Args:
            prompts: Input prompt texts
            completions: Model-generated completion texts
            ground_truth: Reference diagnoses (e.g., "[肺部感染]")

        Returns:
            List of reward scores in [0.0, 1.0]
        """
        if ground_truth is None:
            ground_truth = [""] * len(completions)

        rewards = []
        for prompt, completion, gt in zip(prompts, completions, ground_truth):
            r_dx = self._compute_dx_reward(completion, gt)
            r_graph = self._compute_graph_reward(completion)
            r_path = self._compute_path_reward(completion)
            total = (self.lambda_dx * r_dx +
                     self.lambda_graph * r_graph +
                     self.lambda_path * r_path)
            rewards.append(total)

        return rewards

    # ═══════════════════════════════════════════════════════════════
    # R_dx: Diagnosis Correctness Reward
    # ═══════════════════════════════════════════════════════════════

    def _compute_dx_reward(self, completion: str, ground_truth: str) -> float:
        """Score diagnostic correctness with partial-credit support.

        Uses a 3-level matching:
          1. Exact match: F1 score between predicted and reference diagnosis sets
          2. KG-proximity match: partial credit if predicted entity is KG-nearby to
             the reference entity (e.g., "肺炎" vs "肺部感染")
          3. String overlap: partial credit for substring/superstring overlaps
        """
        pred_dx = self.extract_diagnoses(completion)
        ref_dx = self.extract_diagnoses(ground_truth)

        if not pred_dx:
            return 0.0

        if not ref_dx:
            return 0.5  # neutral when no reference

        # Level 1: Exact match F1
        tp = len(set(pred_dx) & set(ref_dx))
        fp = len(set(pred_dx) - set(ref_dx))
        fn = len(set(ref_dx) - set(pred_dx))

        exact_score = 0.0
        if tp > 0:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            exact_score = 2 * precision * recall / max(precision + recall, 1e-8)

        # Level 2: KG-proximity bonus for non-exact matches
        kg_bonus = 0.0
        for pred in set(pred_dx) - set(ref_dx):
            best_sim = 0.0
            pred_ids = self.entity_names.get(pred, [])
            for ref in set(ref_dx) - set(pred_dx):
                ref_ids = self.entity_names.get(ref, [])
                for pi in pred_ids:
                    for ri in ref_ids:
                        # Check 1-hop connection via cache
                        if self._has_kg_connection(pi, ri, max_hops=1):
                            best_sim = max(best_sim, 0.7)
                        elif self._has_kg_connection(pi, ri, max_hops=2):
                            best_sim = max(best_sim, 0.4)
            kg_bonus += best_sim * 0.3  # Dampen KG bonus

        # Level 3: String overlap bonus
        str_bonus = 0.0
        for pred in set(pred_dx) - set(ref_dx):
            for ref in set(ref_dx) - set(pred_dx):
                if pred in ref or ref in pred:
                    str_bonus += 0.2
                    break

        # Combine
        score = exact_score + min(kg_bonus, 0.3) + min(str_bonus, 0.2)
        return min(score, 1.0)

    @staticmethod
    def extract_diagnoses(text: str) -> List[str]:
        """Extract diagnosis labels from text.

        Handles multiple formats:
          - [肺部感染]
          - [疾病1],[疾病2]
          - 最终答案：[疾病1],[疾病2]
          - 肺部相关诊断：[疾病1],[疾病2]
          - 肺部相关疾病诊断：疾病1,疾病2
          - 诊断：疾病1
        """
        # Strip markdown formatting (**bold**, __underline__, *italic*)
        clean_text = re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text)
        clean_text = re.sub(r'_{1,2}([^_]+?)_{1,2}', r'\1', clean_text)

        diagnoses = []

        # Pattern 1: [...] bracket notation (most reliable, from ground truth)
        bracket_pattern = re.findall(r'\[([^\]]+)\]', clean_text)
        for match in bracket_pattern:
            for item in re.split(r'[,，、;；]', match):
                item = KGRewardFunction._clean_diagnosis(item)
                if item:
                    diagnoses.append(item)

        # Pattern 2: Label-prefix extraction
        # Ordered by specificity: "最终答案" > "肺部相关疾病诊断" > "肺部相关诊断" > "诊断"
        label_patterns = [
            r'最终答案[：:]\s*([^\n]+)',
            r'肺部相关疾病诊断[：:]\s*([^\n]+)',
            r'肺部相关诊断[：:]\s*([^\n]+)',
            r'诊断[：:]\s*([^\n]+)',
        ]
        for pattern in label_patterns:
            for match in re.findall(pattern, clean_text):
                for item in re.split(r'[,，、;；]', match):
                    item = KGRewardFunction._clean_diagnosis(item)
                    if item:
                        diagnoses.append(item)

        # Deduplicate while preserving order
        seen = set()
        unique_dx = []
        for d in diagnoses:
            if d not in seen:
                seen.add(d)
                unique_dx.append(d)

        return unique_dx

    @staticmethod
    def _clean_diagnosis(item: str) -> str:
        """Clean a single diagnosis label.

        - Strips whitespace, brackets [], and formatting punctuation
        - Removes trailing parenthetical content: 肺部感染（尤其是肺炎）→ 肺部感染
        - Removes markdown/formatting residues
        - Filters out obviously non-diagnosis text
        """
        # Strip brackets, whitespace, and formatting chars
        item = item.strip().strip('*_-#。，, []')

        if not item or len(item) < 2:
            return ""

        # Remove trailing parenthetical explanations (full and partial)
        # Full:  肺部感染（尤其是肺炎）→ 肺部感染
        # Partial: 终末期肺病（如慢性阻塞性... → 终末期肺病
        item = re.sub(r'[（(][^)）]*[)）]?$', '', item).strip()

        # Remove leading "如" or "比如" (exemplification)
        item = re.sub(r'^(?:如|比如|例如|如为)\s*', '', item).strip()

        # Strip brackets again after parenthetical removal
        item = item.strip().strip('*_-#。，, []')

        # Filter non-diagnosis text
        skip_starts = ('请', '建议', '注意', '以上', '综上', '目前', '考虑', '当前',
                       '该病', '该患', '总', '综', '因此', '所以', '依据',
                       '首先', '其次', '然后', '最后', '另外', '此外',
                       '最终答案', '肺部相关', '诊断')
        skip_contains = ('以下几种', '包括但不限于', '考虑为', '不排除', '需排除',
                         '需进一步', '建议进一步', '必要时', '应进一步',
                         '综上所述', '结合以上', '综合以上')
        if any(item.startswith(w) for w in skip_starts):
            return ""
        if any(w in item for w in skip_contains):
            return ""

        # Filter items that are just parenthetical leftovers
        if item.endswith('）') or item.endswith(')') or item.startswith('（') or item.startswith('('):
            return ""

        if len(item) < 2:
            return ""

        return item

    # ═══════════════════════════════════════════════════════════════
    # R_graph: Graph Faithfulness Reward
    # ═══════════════════════════════════════════════════════════════

    def _compute_graph_reward(self, completion: str) -> float:
        """Score how well medical claims in the completion are supported by the KG.

        Extracts medical entity mentions, forms entity pairs, and checks
        whether each pair has a valid KG edge (full support) or at least
        co-exists as nodes (partial support).
        """
        # Extract entity mentions from the completion
        mentioned_entities = self._extract_entity_mentions(completion)

        if len(mentioned_entities) < 2:
            # No entity pairs to evaluate
            return 0.0

        # Form entity pairs that co-occur in the same "claim" (sentence)
        sentences = re.split(r'[。！？\n]', completion)
        n_sup = 0
        n_partial = 0
        n_claims = 0

        for sentence in sentences:
            sentence_entities = self._extract_entity_mentions(sentence)
            if len(sentence_entities) < 2:
                continue
            n_claims += 1

            # Check pairs in this claim
            pairs_supported = 0
            pairs_total = 0
            for i in range(len(sentence_entities)):
                for j in range(i + 1, len(sentence_entities)):
                    pairs_total += 1
                    entities_i = self.entity_names.get(sentence_entities[i], [])
                    entities_j = self.entity_names.get(sentence_entities[j], [])

                    # Check if any pair has a KG edge
                    has_edge = False
                    for ni in entities_i:
                        for nj in entities_j:
                            if self._has_kg_connection(ni, nj):
                                has_edge = True
                                break
                        if has_edge:
                            break

                    if has_edge:
                        pairs_supported += 1

            if pairs_total > 0:
                claim_support_ratio = pairs_supported / pairs_total
                if claim_support_ratio >= 0.5:
                    n_sup += 1
                elif claim_support_ratio > 0:
                    n_partial += 1

        if n_claims == 0:
            n_claims = max(len(mentioned_entities) - 1, 1)
            # Fallback: check if any entity pair in the whole text is KG-connected
            connected_pairs = 0
            total_pairs = 0
            for i in range(len(mentioned_entities)):
                for j in range(i + 1, len(mentioned_entities)):
                    total_pairs += 1
                    entities_i = self.entity_names.get(mentioned_entities[i], [])
                    entities_j = self.entity_names.get(mentioned_entities[j], [])
                    for ni in entities_i:
                        for nj in entities_j:
                            if self._has_kg_connection(ni, nj):
                                connected_pairs += 1
                                break
                        else:
                            continue
                        break
            if total_pairs > 0:
                ratio = connected_pairs / total_pairs
                n_sup = ratio * n_claims
                n_partial = (1 - ratio) * n_claims * 0.5

        return (n_sup + 0.5 * n_partial) / max(n_claims, 1)

    def _extract_entity_mentions(self, text: str) -> List[str]:
        """Extract KG entity names from text using dictionary matching.

        Uses longest-match-first to avoid partial/overlapping matches.
        Only returns entities whose names are at least 2 Chinese characters.
        """
        if not text:
            return []

        found = set()
        # Sorted by length desc for longest-match-first
        for entity_name in self.all_entity_names:
            if len(entity_name) < 2:
                continue
            if entity_name in text:
                # Check if this entity is not a substring of an already-found entity
                is_subsumed = False
                for existing in found:
                    if entity_name in existing and entity_name != existing:
                        is_subsumed = True
                        break
                if not is_subsumed:
                    found.add(entity_name)

        return list(found)

    # ═══════════════════════════════════════════════════════════════
    # R_path: Relation/Path Consistency Reward
    # ═══════════════════════════════════════════════════════════════

    def _compute_path_reward(self, completion: str) -> float:
        """Score how well the reasoning uses valid KG relation types.

        Extracts statements about entity relationships from the completion
        and checks whether the implied relation type matches a valid KG
        relation type.
        """
        # Extract entity-relation-entity patterns
        path_triples = self._extract_relation_paths(completion)

        n_path = len(path_triples)
        if n_path == 0:
            return 0.0

        n_valid = 0
        for triple in path_triples:
            subj, rel_indicator, obj = triple
            # Check if relation indicator matches a valid KG relation type
            if self._is_valid_relation(subj, obj, rel_indicator):
                n_valid += 1

        return n_valid / max(n_path, 1)

    def _extract_relation_paths(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract implied (entity, relation_indicator, entity) triples from text.

        Stricter extraction: only returns entity pairs that appear in the same
        sentence with a clear relational indicator between them.
        """
        triples = []
        entities_in_text = self._extract_entity_mentions(text)

        if len(entities_in_text) < 2:
            return []

        # Relation indicators that suggest a medical relationship
        relation_indicators = [
            '引起', '导致', '表现', '症状', '检查', '诊断', '治疗',
            '并发症', '提示', '显示', '支持', '排除',
            '常见于', '多见于', '提示为', '考虑', '怀疑',
            '属于', '包括', '分为', '表现为', '出现',
            '使用', '给予', '予以', '服用', '注射',
            '发现', '检出', '检测', '测定',
            '提示', '符合', '支持', '依据',
        ]

        sentences = re.split(r'[。！？\n；;]', text)
        for sentence in sentences:
            sent_entities = self._extract_entity_mentions(sentence)
            if len(sent_entities) < 2:
                continue

            # Only consider sentences with explicit relational indicators
            found_indicators = [kw for kw in relation_indicators if kw in sentence]
            if not found_indicators:
                continue

            # Pair entities within the sentence, checking for interleaving relation words
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    ei, ej = sent_entities[i], sent_entities[j]
                    pos_i = sentence.find(ei)
                    pos_j = sentence.find(ej)
                    if pos_i < 0 or pos_j < 0:
                        continue

                    # Find if any relation indicator sits between or near the two entities
                    for kw in found_indicators:
                        pos_kw = sentence.find(kw)
                        if pos_i < pos_kw < pos_j or pos_j < pos_kw < pos_i:
                            triples.append((ei, kw, ej))
                            break
                    else:
                        # Also accept if both entities are very close (<50 chars apart)
                        if abs(pos_i - pos_j) < 50:
                            for kw in found_indicators:
                                pos_kw = sentence.find(kw)
                                if abs(pos_kw - min(pos_i, pos_j)) < 50:
                                    triples.append((ei, kw, ej))
                                    break

        return triples

    def _is_valid_relation(self, entity1: str, entity2: str,
                           rel_indicator: str) -> bool:
        """Check if the relation between two entities is valid in the KG.

        Verifies: (1) the relation indicator matches a valid KG relation type,
        and (2) the entity pair has a connecting edge in the KG.
        """
        # Check 1: Relation indicator matches valid KG relation types
        indicator_valid = False
        for valid_rel in self.valid_relation_types:
            if rel_indicator in valid_rel or valid_rel in rel_indicator:
                indicator_valid = True
                break
        if not indicator_valid:
            return False

        # Check 2: Entity pair has a KG edge (1-hop or 2-hop)
        ids1 = self.entity_names.get(entity1, [])
        ids2 = self.entity_names.get(entity2, [])

        for n1 in ids1:
            for n2 in ids2:
                if self._has_kg_connection(n1, n2, max_hops=2):
                    return True

        # Even if entity pair not found in KG, still give credit if
        # the relation indicator is a valid KG relation type
        return False  # Strict: require entity pair KG support

    # ═══════════════════════════════════════════════════════════════
    # Utility
    # ═══════════════════════════════════════════════════════════════

    def get_component_rewards(
        self, completion: str, ground_truth: str
    ) -> Dict[str, float]:
        """Get individual reward components for debugging."""
        return {
            "r_dx": self._compute_dx_reward(completion, ground_truth),
            "r_graph": self._compute_graph_reward(completion),
            "r_path": self._compute_path_reward(completion),
        }


# ── Module-level test ──
if __name__ == "__main__":
    import os
    import sys
    reward_fn = KGRewardFunction(
        nodes_path=os.path.join(os.path.dirname(__file__), "kg_data", "nodes.csv"),
        edges_path=os.path.join(os.path.dirname(__file__), "kg_data", "edges.csv"),
    )

    # Test with sample completions
    test_cases = [
        (
            "请阅读下面的病历，并完成肺部疾病诊断。\n病历：患者咳嗽、咳痰3天，伴发热...",
            "诊断推理：患者有咳嗽、咳痰、发热等呼吸道感染典型症状...\n\n肺部相关诊断：[肺部感染]",
            "[肺部感染]",
        ),
        (
            "请阅读下面的病历...",
            "诊断推理：患者表现为胸闷、呼吸困难，CT提示双肺透光度增强，肺纹理紊乱，考虑慢性支气管炎可能。症状如呼吸急促在急性支气管炎中常见。\n\n肺部相关诊断：[慢性支气管炎],[肺气肿]",
            "[慢性支气管炎],[肺气肿]",
        ),
        (
            "请阅读下面的病历...",
            "不知道，看不懂。",
            "[肺部感染]",
        ),
    ]

    for i, (prompt, completion, gt) in enumerate(test_cases):
        components = reward_fn.get_component_rewards(completion, gt)
        total = reward_fn([prompt], [completion], [gt])[0]
        print(f"\nTest {i}: total={total:.3f}, {components}")
