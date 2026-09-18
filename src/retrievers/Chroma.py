from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import ConfigDict, Field
from tqdm import tqdm


class VectorRetriever(BaseRetriever):

    vectorstore: Chroma = Field(description="Chroma vectorstore")
    k: int = Field(default=5, description="Number of results")
    embeddings: HuggingFaceEmbeddings | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def index(cls, documents: list[Document],
              embeddings: HuggingFaceEmbeddings, k: int = 5,
              path: str = 'data/processed/vector') -> "VectorRetriever":
        """Builds a Chroma index from documents and returns a retriever"""

        first_batch = documents[:250]

        vectorstore = Chroma.from_documents(
            documents=first_batch, embedding=embeddings,
            persist_directory=path, collection_name="rag_chunks")

        remaining_docs = documents[250:]
        if remaining_docs:
            for i in tqdm(range(0, len(remaining_docs), 250), desc='Indexing',
                          colour='cyan'):
                batch = remaining_docs[i: i + 250]
                vectorstore.add_documents(batch)

        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    @classmethod
    def from_index(cls, path: str,
                   embeddings: HuggingFaceEmbeddings | None,
                   k: int = 5) -> "VectorRetriever":
        """Loads an existing Chroma colletion from disk and
           returns a retriever"""
        vectorstore = Chroma(persist_directory=path,
                             embedding_function=embeddings,
                             collection_name="rag_chunks")
        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
        """Gets the top-k documents matching the query using
           similarity, and returns it"""
        result = self.vectorstore.similarity_search_with_score(query, k=self.k)
        output = []
        for doc, score in result:
            improved_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata,
                          "chroma_score": round(float(score), 4)})
            output.append(improved_doc)
        return output
