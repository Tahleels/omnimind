"""
test_dataset.py — Curated Evaluation Dataset

20 question-answer pairs about LangChain concepts.
These are used to measure the quality of the RAG pipeline with RAGAS.

Ground truth answers are concise reference answers used for comparing
against the model's generated output.
"""

EVAL_DATASET = [
    {
        "question": "What is LCEL and what is it used for in LangChain?",
        "ground_truth": "LCEL (LangChain Expression Language) is a declarative way to compose chains in LangChain. It allows you to build complex pipelines by chaining together runnables using the pipe operator (|). It supports streaming, async execution, and parallel execution out of the box.",
    },
    {
        "question": "What is the difference between a ChatModel and an LLM in LangChain?",
        "ground_truth": "In LangChain, an LLM takes a string as input and returns a string. A ChatModel takes a list of messages as input and returns a message. ChatModels support roles like system, human, and AI, making them better suited for conversational applications.",
    },
    {
        "question": "How does RecursiveCharacterTextSplitter work?",
        "ground_truth": "RecursiveCharacterTextSplitter splits text by trying a list of separators in order (e.g., double newline, single newline, space) and recursively splits chunks that are still too large. It tries to keep semantically related text together by preferring higher-level separators first.",
    },
    {
        "question": "What are vector stores used for in RAG?",
        "ground_truth": "Vector stores are databases that store embedding vectors and enable efficient similarity search. In RAG, they are used to store embedded document chunks and retrieve the most semantically similar chunks for a given query.",
    },
    {
        "question": "What is a retriever in LangChain?",
        "ground_truth": "A retriever is an interface that returns documents given an unstructured query. Unlike vector stores, retrievers do not need to store documents themselves. They expose a get_relevant_documents method and can wrap vector stores, BM25 indexes, or other search backends.",
    },
    {
        "question": "What are prompt templates in LangChain and why use them?",
        "ground_truth": "Prompt templates are reusable templates for generating prompts dynamically. They accept input variables and format them into a final prompt string or message list. They promote reusability, testability, and separation of concerns in LangChain applications.",
    },
    {
        "question": "How do output parsers work in LangChain?",
        "ground_truth": "Output parsers transform the raw string output of an LLM into structured formats like JSON, lists, Pydantic objects, or Python dicts. They are chained after an LLM in an LCEL pipeline using the pipe operator.",
    },
    {
        "question": "What is the purpose of memory in LangChain agents and chains?",
        "ground_truth": "Memory allows LangChain chains and agents to persist information between interactions. It enables multi-turn conversations by storing and retrieving conversation history, summaries, or specific entities from previous turns.",
    },
    {
        "question": "What are tools in LangChain and how do agents use them?",
        "ground_truth": "Tools are functions or APIs that agents can call to interact with the outside world (e.g., web search, calculators, databases). Agents use an LLM to reason about which tool to call and with what inputs, and then act on the tool's output.",
    },
    {
        "question": "What is an agent in LangChain?",
        "ground_truth": "An agent is a system that uses an LLM as a reasoning engine to decide which actions to take. It iterates between thinking (using the LLM) and acting (using tools) until it arrives at a final answer. LangGraph is commonly used to build production-grade agents.",
    },
    {
        "question": "What is the difference between embedding models and chat models?",
        "ground_truth": "Embedding models convert text into numerical vector representations for similarity search. Chat models generate text responses given a conversation. Embedding models are used in the retrieval step; chat models are used in the generation step of a RAG pipeline.",
    },
    {
        "question": "What are document loaders in LangChain?",
        "ground_truth": "Document loaders are interfaces for loading documents from various sources such as PDFs, websites, databases, and APIs into LangChain's Document format, which includes page_content and metadata fields.",
    },
    {
        "question": "How does streaming work in LangChain chains?",
        "ground_truth": "Streaming in LangChain allows tokens to be yielded incrementally as the LLM generates them, rather than waiting for the full response. LCEL chains support streaming via the .stream() method, which returns an iterator of chunks.",
    },
    {
        "question": "What is a runnable in LangChain?",
        "ground_truth": "A Runnable is the base interface for all composable components in LangChain. It defines methods like invoke, stream, batch, and their async variants. All chains, prompts, LLMs, and output parsers implement the Runnable interface, enabling composition via LCEL.",
    },
    {
        "question": "What is LangGraph and how is it different from LangChain?",
        "ground_truth": "LangGraph is a library built on top of LangChain for creating stateful, multi-step AI applications using a graph structure of nodes and edges. While LangChain focuses on composing simple linear chains, LangGraph supports cycles, conditionals, and persistent state, making it ideal for complex agentic workflows.",
    },
    {
        "question": "How can you do batch inference with LangChain?",
        "ground_truth": "LangChain runnables support batch inference via the .batch() method, which processes a list of inputs concurrently. This is more efficient than calling .invoke() in a loop as it can parallelize API calls.",
    },
    {
        "question": "What is RAG and how does LangChain support it?",
        "ground_truth": "RAG (Retrieval-Augmented Generation) enhances LLM responses by grounding them in retrieved documents. LangChain supports RAG through document loaders, text splitters, embedding models, vector stores, retrievers, and LCEL chains that combine retrieval with generation.",
    },
    {
        "question": "What is the ChatPromptTemplate and how is it different from PromptTemplate?",
        "ground_truth": "ChatPromptTemplate formats a list of messages for chat models, supporting roles like system, human, and AI. PromptTemplate formats a single string for text-completion LLMs. Use ChatPromptTemplate for chat models and PromptTemplate for older text LLMs.",
    },
    {
        "question": "What are the main text splitter strategies available in LangChain?",
        "ground_truth": "LangChain offers several text splitters: RecursiveCharacterTextSplitter (splits by character hierarchy), CharacterTextSplitter (splits on a single separator), TokenTextSplitter (splits by token count), and language-specific splitters for code (Python, JS, etc.).",
    },
    {
        "question": "How does async support work in LangChain?",
        "ground_truth": "All LangChain runnables support async execution through async variants of their methods: ainvoke, astream, and abatch. These methods use Python's asyncio and are useful for building high-throughput servers that handle many concurrent requests without blocking.",
    },
]
