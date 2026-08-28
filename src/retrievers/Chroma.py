from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


class VectorRetriever(BaseRetriever):

    vectorstore: Chroma = Field(description="Chroma vectorstore")
    k: int = Field(default=5, description="Number of results")
    embeddings: HuggingFaceEmbeddings | None = Field(default=None, description="Embedding model")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def index(self, documents: list[Document], k: int = k, path: str = 'data/processed/vector') -> VectorRetriever:
        pass
