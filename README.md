# langchain-superlocalmemory

LangChain integration for [SuperLocalMemory V3](https://superlocalmemory.com) — local-first AI memory with mathematical foundations.

## Features

- **`SuperLocalMemoryChatHistory`** — Drop-in `BaseChatMessageHistory` for conversation persistence across sessions
- **`SuperLocalMemoryRetriever`** — `BaseRetriever` for RAG-style memory augmentation with 4-channel retrieval
- **Direct Python API** — No subprocess calls, no CLI dependency at runtime
- **Privacy-first** — All data stays on your device (Mode A: zero cloud)

## Installation

```bash
pip install langchain-superlocalmemory
```

Requires SuperLocalMemory V3 to be installed:

```bash
pip install "superlocalmemory[search]"
# or
npm install -g superlocalmemory && slm setup
```

## Usage

### Chat Message History

```python
from langchain_superlocalmemory import SuperLocalMemoryChatHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Create history for a session
history = SuperLocalMemoryChatHistory(session_id="my-project")

# Use with RunnableWithMessageHistory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: SuperLocalMemoryChatHistory(session_id=session_id),
    input_messages_key="input",
    history_messages_key="history",
)
```

### Memory Retriever (RAG)

```python
from langchain_superlocalmemory import SuperLocalMemoryRetriever

retriever = SuperLocalMemoryRetriever(k=5, score_threshold=0.3)
docs = retriever.invoke("authentication middleware patterns")

for doc in docs:
    print(f"{doc.page_content} (score: {doc.metadata['score']:.2f})")
```

### Modes

| Mode | Privacy | Performance | Use Case |
|------|---------|-------------|----------|
| `a` | Maximum (zero cloud) | 74.8% LoCoMo | EU AI Act compliant |
| `b` | High (local Ollama) | Higher | Private + LLM assist |
| `c` | Standard (cloud LLM) | 87.7% LoCoMo | Maximum accuracy |

```python
# EU AI Act compliant (default)
history = SuperLocalMemoryChatHistory(session_id="s1", mode="a")

# Full power
history = SuperLocalMemoryChatHistory(session_id="s1", mode="c")
```

## Links

- [SuperLocalMemory](https://superlocalmemory.com)
- [Paper: arXiv:2603.14588](https://arxiv.org/abs/2603.14588)
- [GitHub](https://github.com/qualixar/superlocalmemory)

## License

MIT

Part of [Qualixar](https://qualixar.com) | Author: [Varun Pratap Bhardwaj](https://varunpratap.com)
