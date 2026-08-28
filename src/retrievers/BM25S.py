import json

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
    documents: list[Document] = Field(default=None)
    corpus: list[str] | None = None

    def index(self, documents: list[Document], k: int = 4,
              path: str | None = None) -> "BM25SRetriever":
        self.corpus = [doc.page_content for doc in documents]
        tokenized = bm25s.tokenize(self.corpus)
        self.retriever = bm25s.BM25()
        try:
            self.retriever.index(tokenized)
        except ValueError:
            raise RetrieverError('[ERROR]: Cannot index corpus')
        if path:
            self.retriever.save(path, corpus=self.corpus)
        return self.retriever

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
        tokenized_query = bm25s.tokenize([query])
        n = min(self.k, len(self.documents))
        results, _ = self.retriever.retrieve(tokenized_query, k=n)
        documents = []
        for i in range(results.shape[1]):
            idx = int(results[0, i])
            documents.append(self.documents[idx])
        return documents

    def save(self, path: str = "./data/processed") -> None:
        try:
            self.retriever.save(path)
            with open(f"{path}/corpus.json", 'w') as f:
                json.dump(self.corpus, f)
            print('[test]')
            docs = [doc.model_dump() for doc in self.documents]
            with open(f"{path}/documents.json", 'w') as f:
                json.dump(docs, f)
        except PermissionError:
            raise RetrieverError("[ERROR]: Cannot save documents in folder -"
                                 " permission denied")

    # def load(self, path: str = "./data/processed") -> None:
