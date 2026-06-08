# graphcot: Architecture and Algorithm Design Analysis

## 1. Data Flow Pipeline (Data Flow)

The data flow of graphcot is a clear unidirectional pipeline, where data passes through four core files in sequence and undergoes transformation:

1. **Data Ingestion** -- `storage.py`
   - **Input**: Local graph database export files `nodes.csv` and `edges.csv`.
   - **Processing**: `SimpleGraphStorage` reads CSV files and converts them into an in-memory NetworkX graph structure for subsequent graph traversal and degree calculation.
2. **Graph Partitioning** -- `graphcot.py`
   - **Input**: The complete in-memory NetworkX large graph.
   - **Processing**: `LongTailPartitioner` intervenes. It does not randomly cut the graph; instead, it algorithmically selects a series of specific "Anchor Nodes" and expands outward from them to find neighbors, forming small **knowledge subgraphs (Communities)**.
3. **Prompt Construction and Generation** -- `graphcot.py` & `llm_client.py`
   - **Input**: Knowledge subgraphs (Communities) partitioned in the previous step.
   - **Processing**: `CoTGenerator` serializes the subgraph structure into text context readable by LLMs, concatenates it with a specific Prompt, and sends it asynchronously to the LLM API via `llm_client.py`.
   - **Interaction**: The LLM performs multi-hop reasoning based on the prompt and returns an XML-formatted string containing `<qa>`, `<question>`, `<thought>`, and `<answer>` tags.
4. **Parsing and Persistence** -- `main.py`
   - **Input**: The XML string returned by the LLM.
   - **Processing**: The system parses the XML, extracts Q&A and chain-of-thought, and uses MD5 hashing to deduplicate questions.
   - **Output**: Finally, a structured `generated_qa.json` dataset is generated under the `output` directory.

---

## 2. Core Algorithm Analysis (Algorithms)

The core technical innovations of this project are concentrated in `graphcot.py`, which mainly contains two algorithms:

### 1. Inverse-Degree Long-Tail Subgraph Sampling Algorithm
Traditional random walks or random sampling tend to oversample densely connected "popular entities" in the graph (e.g., common disease "cold"), while marginal "long-tail entities" (e.g., rare diseases or new drugs) are ignored.

**Algorithm Flow**:
- **Degree Calculation**: Compute the degree (number of connections) of all nodes in the graph. Lower degree indicates the entity is more "long-tail".
- **Probability Mapping**: Use the inverse-degree probability formula (e.g., $P(v) \propto \frac{1}{\text{degree}(v) + \epsilon}$) to assign each node a probability of being selected as an Anchor. The selection probability of long-tail entities is significantly amplified.
- **Adaptive BFS Expansion**: After selecting a long-tail entity, use it as the starting point for breadth-first search (BFS). The algorithm controls the expansion depth (e.g., within 2 hops) and the maximum number of nodes in the subgraph (e.g., no more than 50 nodes), ensuring that the partitioned subgraph contains both long-tail knowledge and sufficient context to support complex Q&A.

### 2. Multi-Hop Chain-of-Thought (CoT) Generation Algorithm
LLMs are prone to hallucination, and directly generating Q&A often lacks logical structure. This project employs a strictly constrained CoT algorithm.

**Algorithm Flow**:
- **Knowledge Injection**: Format node attributes (entity types, descriptions) and edge relations (A treats B, B causes C) from the subgraph into rigorous assertions (Facts).
- **Instruction Constraint**: Force the LLM to generate `<thought>` tags before answering in the System Prompt.
- **Logic Validation**: The LLM must write in `<thought>`: "Step 1: Based on known information... Step 2: Combining... Step 3: Draw conclusions..." before finally outputting `<answer>`. This mechanism greatly reduces knowledge hallucination and endows the generated data with high logical quality.

---

## 3. Core Innovations

Compared with traditional graph-based QA generation tools, graphcot achieves the following innovations:

1. **Long-Tail Preference to Address "Knowledge Collapse"**
   Through the innovative "LongTailPartitioner," it solves the problem of homogeneous data generated from graphs. It actively mines sparse knowledge at the edges of the graph, producing Q&A with broader coverage, making it highly suitable for improving vertical LLMs' ability to handle rare long-tail problems (e.g., rare complications in medicine).

2. **Native Chain-of-Thought Data Engine**
   The current open-source landscape is severely lacking high-quality datasets with reasoning processes. This project goes beyond merely generating Q&A by directly mapping graph structures to logical reasoning chains (Path -> Reasoning). The generated data is naturally suited for training models focused on logical reasoning like DeepSeek-R1.

3. **High-Concurrency and Asynchronous Decoupled Architecture**
   Through `llm_client.py`, it implements fully asynchronous (`asyncio` + `aiohttp`) API calls, supporting local multi-coroutine concurrent processing of multiple subgraphs, resulting in several times faster generation. Additionally, the codebase is highly concentrated within 4 files, making it extremely easy for secondary development and deployment.


``` bash
cd graphcot
nohup python main.py \
  --kg_dir ../LungKG/LungKG_fusion/output \
  --output_dir output_optimal \
  --max_communities 8000 \
  --num_questions 4 \
  --tail_ratio 0.75 \
  --concurrency 2 \
  > generate_qa.log 2>&1 &
```
