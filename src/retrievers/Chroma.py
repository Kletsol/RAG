from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


class VectorRetriever(BaseRetriever):

    vectorstore: Chroma = Field(description="Chroma vectorstore")
    k: int = Field(default=5, description="Number of results")
    embeddings: HuggingFaceEmbeddings | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def index(cls, documents: list[Document],
              embeddings: HuggingFaceEmbeddings, k: int = 5,
              path: str = 'data/processed/vector') -> "VectorRetriever":
        first_batch = documents[:250]
        vectorstore = Chroma.from_documents(
            documents=first_batch, embedding=embeddings,
            persist_directory=path, collection_name="test")
        remaining_docs = documents[250:]
        if remaining_docs:
            for i in range(0, len(remaining_docs), 250):
                batch = remaining_docs[i: i + 250]
                vectorstore.add_documents(batch)
        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    @classmethod
    def from_index(cls, path: str,
                   embeddings: HuggingFaceEmbeddings | None,
                   k: int = 5) -> "VectorRetriever":
        embeddings = embeddings  # noqa: PLW0127
        vectorstore = Chroma(persist_directory=path,
                             embedding_function=embeddings,
                             collection_name="test")
        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
        result = self.vectorstore.similarity_search_with_score(query, k=self.k)
        output = []
        for doc, score in result:
            improved_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata,
                          "chroma_score": round(float(score), 4)})
            output.append(improved_doc)
        return output
