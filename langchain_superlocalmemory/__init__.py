"""LangChain integration for SuperLocalMemory V3.

Provides:
- SuperLocalMemoryChatHistory: BaseChatMessageHistory for conversation persistence
- SuperLocalMemoryRetriever: BaseRetriever for RAG-style memory augmentation

Part of Qualixar | Author: Varun Pratap Bhardwaj (qualixar.com | varunpratap.com)
Paper: https://arxiv.org/abs/2603.14588
"""

from langchain_superlocalmemory.chat_history import SuperLocalMemoryChatHistory
from langchain_superlocalmemory.retriever import SuperLocalMemoryRetriever

__all__ = [
    "SuperLocalMemoryChatHistory",
    "SuperLocalMemoryRetriever",
]

__version__ = "0.1.1"
