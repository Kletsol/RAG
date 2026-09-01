import json
from pathlib import Path

import bm25s
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


class RetrieverError(Exception):
    pass


class BM25SRetriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: bm25s.BM25 = Field(default=None)
    documents: list[Document] = Field(default_factory=list)
    k: int = Field(default=5)

    @classmethod
    def index(cls, documents: list[Document], k: int = 5,
              path: str | None = None) -> "BM25SRetriever":
        if not documents:
            raise RetrieverError("[ERROR]: Cannot index empty corpus")
        corpus = [doc.page_content for doc in documents]
        tokenized = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        try:
            retriever.index(tokenized)
        except ValueError as e:
            raise RetrieverError('[ERROR]: Cannot index corpus') from e
        if path:
            retriever.save(path, corpus=corpus)
        return cls(retriever=retriever, documents=documents, k=k)

    @classmethod
    def from_index(cls, path, documents: list[Document], k: int = 5
                   ) -> "BM25SRetriever":
        try:
            index = bm25s.BM25.load(path, load_corpus=False)
        except (FileNotFoundError, ValueError) as e:
            raise RetrieverError("[ERROR]: Cannot load BM25 index") from e
        return cls(retriever=index, documents=documents, k=k)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
        if not self.documents:
            return []
        tokenized_query = bm25s.tokenize([query])
        n = min(self.k, len(self.documents))
        results, scores = self.retriever.retrieve(tokenized_query, k=n)
        documents = []
        for i in range(results.shape[1]):
            index = int(results[0, i])
            document = self.documents[index]
            document = Document(page_content=document.page_content,
                                metadata={**document.metadata,
                                          "bm25_score": float(scores[0, i])})
            documents.append(document)
        return documents

    def save(self, path: str = "./data/processed") -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            self.retriever.save(str(directory))
            documents = [document.model_dump() for document in self.documents]
            with open(directory / "documents.json", 'w', encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
        except PermissionError as e:
            raise RetrieverError("[ERROR]: Cannot save documents in folder -"
                                 " permission denied") from e

    # def load(self, path: str = "./data/processed") -> None:
