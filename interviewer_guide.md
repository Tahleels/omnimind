# OmniMind — Interviewer Guide 🤓

This guide explains the architecture, features, and design choices of **OmniMind (RAG v2)** in a simple, clear way without heavy academic jargon. It is designed to help you explain the engineering value of the codebase during interviews.

---

## 🌟 The Core Pitch: What is OmniMind?
OmniMind is a **Universal Document Intelligence Hub**. Users can upload PDFs, Word files, websites, or YouTube videos, organize them into separate "workspaces," and have intelligent conversations about them.

The key upgrade in **v2** is that it moves from a simple "Search-and-Generate" script to a **stateful agent** built with **LangGraph**. It acts more like a human assistant: it remembers past conversation turns, checks the internet if it doesn't know the answer, uses a hybrid search strategy to find document context, and self-checks its output to make sure it didn't make up any citations.

---

## 🗺️ How a Query Flows (The Pipeline)

When a user types a message, it travels through a **directed graph** managed by LangGraph:

```
  [User Message] 
         │
         ▼
    1. ROUTER (Heuristic Classifier)
         ├─► "chat" ──► 4. GENERATE (AI answers casually) ──────┐
         ├─► "web"  ──► 4. GENERATE (AI searches the internet) ─┼─► 5. VALIDATOR (Stripped links) ──► [User Receives Answer]
         └─► "rag"  ──► 2. RETRIEVE (Vector + BM25 Search)     │
                             │                                   │
                             ▼                                   │
                        3. RERANK (Sorts best hits) ─────────────┘
```

1. **The Router (The Decision Maker):** 
   Instead of using an expensive AI call to decide what to do, we use fast text-pattern matching (regex). 
   * If the user says "Hi", it routes to `chat` mode (casual talk).
   * If the user asks for news or mentions a URL, it routes to `web` mode (live search).
   * Otherwise, it defaults to `rag` mode (document search).

2. **The Retrievers (The Library Search):**
   We use a **Hybrid Search** approach:
   * **Semantic Search (Vector):** Finds matches based on *meaning* (using local vector embeddings in Qdrant).
   * **Keyword Search (BM25):** Finds matches based on exact *text matching* (like Ctrl+F).
   * **Reciprocal Rank Fusion (RRF):** Blends the two search lists together based on rank order. It's like asking two librarians for book lists and putting the books that appeared on both lists at the top.

3. **The Reranker (The Quality Filter):**
   Once we have our top search hits, a local machine learning model (**Cross-Encoder**) re-scores them to sort the most relevant paragraphs to the very top. This takes only ~80ms on a standard CPU but significantly boosts answer accuracy.

4. **The Generator (The Writer):**
   We feed the sorted paragraphs, the user's question, and the last 10 messages of **conversation memory** to the Language Model (Gemini/OpenAI) to write a response. If the primary model fails or hits rate limits, the system instantly falls back to a backup model so the user never sees an error.

5. **The Citation Validator (The Guardrail):**
   The AI is instructed to cite its sources using numbers like `[1]` or `[2]`. Before showing the message to the user, our validator program parses the response and strips out any "hallucinated" citations that don't match the source documents we retrieved.

---

## 💡 Smart Engineering Decisions (Ready for scale)

Here are the key technical highlights to talk about in interviews:

*   **Workspace Separation (Filters over Collections):**
    Instead of creating a brand-new Qdrant database collection for every user or workspace (which is slow and resource-heavy), we store all documents in **one** collection but stamp every text chunk with a `workspace_id`. When querying, we tell Qdrant to filter by that ID. This is fast, cheap, and easily scales to thousands of workspaces.
    
*   **Dynamic, Zero-Setup Keyword Search:**
    Keyword search indexes (like BM25) usually need to be pre-calculated and saved on disk. For user-uploaded documents, we build a **dynamic BM25 index on-the-fly** directly from the retrieved Qdrant vectors. This means as soon as a user uploads a PDF, they can search it immediately with both vector and keyword search without reloading the server.

*   **Thread-Safe LRU Query Cache:**
    Identical queries in the same workspace return instantly because we cache the results in memory. The cache is thread-safe and drops the oldest entries when it reaches a limit of 100 queries.

*   **Offline First (Privacy & Cost Savings):**
    Our embedding model (`all-MiniLM-L6-v2`) and reranking model run entirely locally on the CPU. This keeps the server running fast without paying a cent for third-party search APIs.

*   **LLM-as-a-Judge Evaluation (`scripts/evaluate_v2.py`):**
    We test the system by sending a set of test questions, feeding the answers back to another LLM, and scoring them for **faithfulness** (factuality) and **answer relevance**. We have a `pytest` gate that fails if the average scores drop below production thresholds (Faithfulness >= 0.75, Relevance >= 0.70).
