---
title: RAG Patterns Analysis
date: 2026-01-19
source: claude-cookbooks-forked repository
category: Technical Analysis
---

# RAG Patterns Analysis

This document analyzes Retrieval Augmented Generation patterns extracted from the Anthropic claude-cookbooks repository. The patterns progress from basic chunk-based retrieval through advanced techniques like summary indexing, re-ranking, and contextual embeddings. Each pattern includes source references, code examples, and guidance on when to apply the technique.

---

## Key Patterns

### 1. Basic RAG with Chunk-Based Retrieval

The foundation of RAG systems involves chunking documents by heading, embedding each chunk, and using cosine similarity for retrieval. This approach is sometimes called "Naive RAG" in the industry.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/guide.ipynb` (cell 7-8)

```python
def retrieve_base(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result["metadata"]
        context += f"\n{chunk['text']}\n"
    return results, context


def answer_query_base(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query:
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.content[0].text
```

**When to use:**
- Initial prototypes and proof-of-concept systems
- Simple Q&A over well-structured documents
- When latency and cost are primary concerns
- Documents with clear, self-contained sections

---

### 2. VectorDB Class Implementation

The VectorDB class provides an in-memory vector database with Voyage AI embeddings, query caching, and disk persistence. This serves as the foundation for all retrieval patterns.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/evaluation/vectordb.py` (lines 9-89)

```python
class VectorDB:
    def __init__(self, name, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, data):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts = [f"Heading: {item['chunk_heading']}\n\n Chunk Text:{item['text']}" for item in data]
        self._embed_and_store(texts, data)
        self.save_db()

    def _embed_and_store(self, texts, data):
        batch_size = 128
        result = [
            self.client.embed(texts[i : i + batch_size], model="voyage-2").embeddings
            for i in range(0, len(texts), batch_size)
        ]
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data

    def search(self, query, k=3, similarity_threshold=0.75):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)
                if len(top_examples) >= k:
                    break
        self.save_db()
        return top_examples
```

**When to use:**
- Prototyping and development environments
- Small to medium datasets (under 100k chunks)
- When you need query caching for repeated searches
- As a template for production vector database integration

---

### 3. Summary Indexing Pattern

Summary indexing enhances retrieval by generating a concise summary for each chunk and embedding the combination of heading, text, and summary together. This captures the essence of each chunk more effectively.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/guide.ipynb` (cells 19-23)

```python
def generate_summaries(input_file, output_file):
    with open(input_file) as f:
        docs = json.load(f)

    knowledge_base_context = "This is documentation for Anthropic's Claude API."

    summarized_docs = []
    for doc in tqdm(docs, desc="Generating summaries"):
        prompt = f"""
        You are tasked with creating a short summary of the following content.
        Context about the knowledge base: {knowledge_base_context}

        Content to summarize:
        Heading: {doc["chunk_heading"]}
        {doc["text"]}

        Please provide a brief summary in 2-3 sentences. The summary should capture
        the key points and be concise. We will use it as part of our search pipeline.
        """
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        summary = response.content[0].text.strip()
        summarized_doc = {
            "chunk_link": doc["chunk_link"],
            "chunk_heading": doc["chunk_heading"],
            "text": doc["text"],
            "summary": summary,
        }
        summarized_docs.append(summarized_doc)

    with open(output_file, "w") as f:
        json.dump(summarized_docs, f, indent=2)
```

**When to use:**
- Technical documentation with dense, specialized content
- When chunks lack sufficient context in isolation
- Knowledge bases where terminology varies between query and document
- Systems where ingestion time is not a constraint

---

### 4. SummaryIndexedVectorDB Extension

This class extends the base VectorDB to embed the concatenation of chunk heading, text, and summary together for richer vector representations.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/evaluation/vectordb.py` (lines 92-174)

```python
class SummaryIndexedVectorDB:
    def __init__(self, name, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/summary_indexed_vector_db.pkl"

    def load_data(self, data):
        if self.embeddings and self.metadata:
            return
        if os.path.exists(self.db_path):
            self.load_db()
            return

        # Key difference: embed heading + text + summary together
        texts = [
            f"{item['chunk_heading']}\n\n{item['text']}\n\n{item['summary']}" for item in data
        ]
        self._embed_and_store(texts, data)
        self.save_db()

    def search(self, query, k=5, similarity_threshold=0.75):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)
                if len(top_examples) >= k:
                    break
        return top_examples
```

**When to use:**
- Building on the summary indexing pattern
- When you need both rich embeddings and original content access
- Production systems with pre-computed summaries

---

### 5. Re-ranking with Claude for Precision

Re-ranking uses Claude to reassess and reorder initially retrieved documents based on their summaries. This casts a wider net initially (retrieve 20 candidates) then narrows to the most relevant (top 3).

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/guide.ipynb` (cell 30)

```python
def rerank_results(query: str, results: list[dict], k: int = 5) -> list[dict]:
    summaries = []
    for i, result in enumerate(results):
        summary = f"[{i}] Document Summary: {result['metadata']['summary']}"
        summaries.append(summary)
    joined_summaries = "\n\n".join(summaries)

    prompt = f"""
    Query: {query}
    You are about to be given a group of documents, each preceded by its index number.
    Your task is to select only {k} most relevant documents from the list.

    <documents>
    {joined_summaries}
    </documents>

    Output only the indices of {k} most relevant documents in order of relevance,
    separated by commas:
    <relevant_indices>put the numbers here</relevant_indices>
    """

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=50,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "<relevant_indices>"},
        ],
        temperature=0,
        stop_sequences=["</relevant_indices>"],
    )

    response_text = response.content[0].text.strip()
    relevant_indices = []
    for idx in response_text.split(","):
        try:
            relevant_indices.append(int(idx.strip()))
        except ValueError:
            continue

    if len(relevant_indices) == 0:
        relevant_indices = list(range(min(k, len(results))))

    relevant_indices = [idx for idx in relevant_indices if idx < len(results)]
    reranked_results = [results[idx] for idx in relevant_indices[:k]]

    for i, result in enumerate(reranked_results):
        result["relevance_score"] = 100 - i

    return reranked_results


def retrieve_advanced(query: str, db, k: int = 3, initial_k: int = 20):
    initial_results = db.search(query, k=initial_k)
    reranked_results = rerank_results(query, initial_results, k=k)

    new_context = ""
    for result in reranked_results:
        chunk = result["metadata"]
        new_context += f"\n<document>\n{chunk['chunk_heading']}\n\n{chunk['text']}\n</document>\n"

    return reranked_results, new_context
```

**When to use:**
- High-stakes applications requiring maximum precision
- When initial retrieval returns many marginally relevant results
- Complex queries that benefit from semantic understanding of relevance
- Systems where answer quality matters more than latency

---

### 6. Assessment Metrics

The cookbook defines five key metrics for assessing RAG systems, covering both retrieval quality and end-to-end performance.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/guide.ipynb` (cells 11-13)

| Metric | Formula | Purpose |
|--------|---------|---------|
| Precision | True Positives / Total Retrieved | Measures efficiency - how many retrieved chunks are relevant |
| Recall | True Positives / Total Correct | Measures completeness - how many relevant chunks were found |
| F1 Score | 2 * (Precision * Recall) / (Precision + Recall) | Balanced measure between precision and recall |
| MRR | (1/Q) * sum(1/rank_i) | Measures ranking quality - how quickly users find relevant info |
| E2E Accuracy | Correct Answers / Total Questions | LLM-as-judge assessment of final answer quality |

```python
def calculate_mrr(retrieved_links: list[str], correct_links: set[str]) -> float:
    for i, link in enumerate(retrieved_links, 1):
        if link in correct_links:
            return 1 / i
    return 0


def assess_retrieval(retrieval_function, assessment_data, db):
    precisions, recalls, mrrs = [], [], []

    for item in tqdm(assessment_data, desc="Assessing Retrieval"):
        retrieved_chunks, _ = retrieval_function(item["question"], db)
        retrieved_links = [
            chunk["metadata"].get("chunk_link", chunk["metadata"].get("url", ""))
            for chunk in retrieved_chunks
        ]
        correct_links = set(item["correct_chunks"])

        true_positives = len(set(retrieved_links) & correct_links)
        precision = true_positives / len(retrieved_links) if retrieved_links else 0
        recall = true_positives / len(correct_links) if correct_links else 0
        mrr = calculate_mrr(retrieved_links, correct_links)

        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(mrrs) / len(mrrs)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    return avg_precision, avg_recall, avg_mrr, f1, precisions, recalls, mrrs
```

**When to use:**
- Always - assessment is essential for any production RAG system
- Precision matters most when context window is limited
- Recall matters most when coverage is needed
- MRR matters most for user-facing search experiences

---

### 7. Contextual Embeddings for Context-Enriched Chunks

Contextual embeddings solve the problem of isolated chunks by using Claude to generate a brief description that "situates" each chunk within its source document before embedding.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/contextual-embeddings/guide.ipynb` (cells 16-20)

```python
DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


def situate_context(doc: str, chunk: str) -> str:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {"type": "ephemeral"},  # prompt caching
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    },
                ],
            }
        ],
    )
    return response
```

**When to use:**
- Codebases where function/class definitions need file-level context
- Legal or financial documents where section context matters
- Any domain where chunks frequently reference content elsewhere in the document
- When you can afford one-time ingestion costs (prompt caching reduces costs by 60-70%)

---

### 8. BM25 + Semantic Hybrid Search

Hybrid search combines semantic embeddings with BM25 keyword search using Reciprocal Rank Fusion to get the best of both approaches.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/contextual-embeddings/guide.ipynb` (cells 25-27)

```python
class ElasticsearchBM25:
    def __init__(self, index_name: str = "contextual_bm25_index"):
        self.es_client = Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.create_index()

    def search(self, query: str, k: int = 20) -> list[dict]:
        response = self.es_client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            size=k,
        )
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "content": hit["_source"]["content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]


def retrieve_hybrid(query, db, es_bm25, k, semantic_weight=0.8, bm25_weight=0.2):
    num_chunks_to_recall = 150

    # Semantic search
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids = [
        (result["metadata"]["doc_id"], result["metadata"]["original_index"])
        for result in semantic_results
    ]

    # BM25 search
    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_bm25_chunk_ids = [
        (result["doc_id"], result["original_index"]) for result in bm25_results
    ]

    # Reciprocal Rank Fusion
    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
    chunk_id_to_score = {}

    for chunk_id in chunk_ids:
        score = 0
        if chunk_id in ranked_chunk_ids:
            index = ranked_chunk_ids.index(chunk_id)
            score += semantic_weight * (1 / (index + 1))
        if chunk_id in ranked_bm25_chunk_ids:
            index = ranked_bm25_chunk_ids.index(chunk_id)
            score += bm25_weight * (1 / (index + 1))
        chunk_id_to_score[chunk_id] = score

    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(),
        key=lambda x: chunk_id_to_score[x],
        reverse=True
    )

    return sorted_chunk_ids[:k]
```

**When to use:**
- Technical documentation with specific function names or identifiers
- Queries that mix natural language with exact terms
- When semantic search alone misses keyword-specific queries
- Production systems that can support Elasticsearch infrastructure

---

### 9. Three-Level RAG Progression

The cookbook demonstrates a progression from basic to advanced RAG with measured improvements at each level.

**Source:** `/Users/personal/Documents/1._Claude_Code/AI-Agent-Tools/claude-cookbooks-forked/capabilities/retrieval_augmented_generation/guide.ipynb` (full notebook)

| Level | Technique | Avg Precision | Avg Recall | Avg MRR | E2E Accuracy |
|-------|-----------|---------------|------------|---------|--------------|
| 1 | Basic RAG | 0.43 | 0.66 | 0.74 | 71% |
| 2 | Summary Indexing | 0.43 | 0.67 | 0.78 | 78% |
| 3 | Summary + Re-ranking | 0.44 | 0.69 | 0.87 | 81% |

The contextual embeddings notebook shows similar progression:

| Approach | Pass@5 | Pass@10 | Pass@20 |
|----------|--------|---------|---------|
| Baseline RAG | 80.92% | 87.15% | 90.06% |
| + Contextual Embeddings | 88.12% | 92.34% | 94.29% |
| + Hybrid Search (BM25) | 88.86% | 92.31% | 95.23% |
| + Reranking | 92.15% | 95.26% | 97.45% |

**When to use each level:**
- Level 1 (Basic): Prototypes, simple use cases, cost-sensitive applications
- Level 2 (Summary): Production systems with one-time ingestion budget
- Level 3 (Re-ranking): High-stakes applications where precision is critical

---

## Reusable Utilities

| Utility | File | Lines | Purpose |
|---------|------|-------|---------|
| VectorDB | `vectordb.py` | 9-89 | In-memory vector store with caching and persistence |
| SummaryIndexedVectorDB | `vectordb.py` | 92-174 | Extended VectorDB with summary concatenation |
| ContextualVectorDB | `contextual-embeddings/guide.ipynb` | cell 20 | VectorDB with Claude-generated context |
| ElasticsearchBM25 | `contextual-embeddings/guide.ipynb` | cell 27 | BM25 search wrapper for hybrid retrieval |
| assess_retrieval | `guide.ipynb` | cell 13 | Calculates precision, recall, MRR, F1 |
| assess_end_to_end | `guide.ipynb` | cell 13 | LLM-as-judge accuracy assessment |
| calculate_mrr | `guide.ipynb` | cell 13 | Mean Reciprocal Rank calculation |
| rerank_results | `guide.ipynb` | cell 30 | Claude-based re-ranking function |
| situate_context | `contextual-embeddings/guide.ipynb` | cell 18 | Generate chunk context with prompt caching |
| retrieve_advanced | Both notebooks | Various | Hybrid or re-ranked retrieval |

---

## Recommendations

### Start Simple, Measure, Then Optimize

Begin with basic RAG to establish a baseline. The assessment framework in these cookbooks makes it straightforward to measure improvements as you add complexity.

### Choose Techniques Based on Your Constraints

Consider these factors when selecting patterns:

- **Latency budget**: Basic RAG is fastest; re-ranking adds 100-200ms per query
- **Ingestion costs**: Contextual embeddings require one-time processing with Claude
- **Infrastructure**: Hybrid search requires Elasticsearch; basic RAG needs only a vector store
- **Accuracy requirements**: Each level adds 5-10 percentage points of improvement

### Prompt Caching is Essential for Contextual Embeddings

When processing chunks from the same document sequentially, prompt caching reduces costs by 60-70%. Always use the `cache_control` parameter when generating contextual descriptions.

### Assessment Should Mirror Production

Create assessment datasets that reflect real user queries. The cookbooks use:

- Questions requiring synthesis across multiple chunks
- Queries with varying specificity
- Ground truth mappings to correct chunks and answers

### Consider the Full Pipeline

Retrieval improvements only matter if they translate to better final answers. Always measure both retrieval metrics (precision, recall, MRR) and end-to-end accuracy with LLM-as-judge assessment.
