*This project has been created as part of the 42 curriculum by lbonnet*

# RAG against the machine 💭

<span style="color:turquoise">

## 📝 Description
</span>

Text

<span style="color:turquoise">

## 🖥️ Instructions
</span>

This project has a Makefile, allowing you to use different rules serving different purposes:

-> **make install:**
    install the project with all its needed dependencies using uv

-> **make debug:**
    run the main script in debug mode using Python’s built-in debugger

-> **make clean:**
    remove temporary files or caches to keep the project environment clean

-> **make lint:**
    execute flake8 and mypy with mandatory flags

-> **make lint-strict:**
    execute flake8 and mypy -- strict

-> **make run:**
    execute the main script of the project

<span style="color:lightblue">

### ⤵️ Input
</span>

Text

<span style="color:lightblue">

### ⤴️ Output
</span>

Text

<span style="color:turquoise">

## 📚 Resources
</span>

Some articles, references and tutorials I used during the elaboration of this project:

- https://realpython.com/llamaindex-examples/ :  

AI usage :

<span style="color:turquoise">

## 🚀 Additional sections
</span>

### -> System architecture

Text

### -> Chunking strategy

Text

### -> Retrieval Method

Text

### -> Performance analysis

Text

### -> Design Decisions

Text

### -> Challenges faced

Text

### -> Example usage

Text



[BaseModel].model_validate(data) for pydantic validation



[project]
name = "42-rag-2-0"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "accelerate>=1.14.0",
    "bm25s>=0.3.9",
    "colorama>=0.4.6",
    "fastapi[standard]>=0.139.0",
    "fire>=0.7.1",
    "langchain>=1.3.12",
    "langchain-community>=0.4.2",
    "pydantic>=2.13.4",
    "torch>=2.13.0",
    "tqdm>=4.68.4",
    "transformers>=5.13.0",
]




Je travaille sur un projet encaldre dont le but est de recrer un RAG. Mes contraintes sont les suivantes :
- Un dossier de fichiers contient les donnees a indexer. Seuls les fichiers .md, .py et .txt m'interessent
- 2 retrievers, BM25S et Chroma, dont les resultats passeront par un RRF par la suite
- Une gestion des etapes (index, load, retrieve...) via un controller
- Des modeles pydantic obligatoires a respecter :
'
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int

The MinimalSource model represents a single source of information

The UnansweredQuestion and AnsweredQuestion models represent an unanswered question and an answered question

The RagDataset model represents a dataset of RAG questions

The MinimalSearchResults and MinimalAnswer models represent the search results and an answer

The StudentSearchResults and StudentSearchResultsAndAnswer models represent search results and search results with answers
'
J'ai deja ecrit mon LoaderSplitter et mes deux retrievers :
LoaderSplitter :
'
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


class LoaderError(Exception):
    pass


class LoaderSplitter:

    def _add_character_indices(self, chunks: list[Document]) -> list[Document]:
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content)
            chunk.metadata["first_character_index"] = start
            chunk.metadata["last_character_index"] = end
        return chunks

    def load_from_extension(self, chunk_size: int, overlap: int, ext: str,
                            path: str = './data/raw') -> list[Document]:
        loader = DirectoryLoader(path, glob=f"**/*.{ext}",
                                 loader_cls=TextLoader)
        splitters = {'md': self.markdown_splitter,
                     'py': self.python_splitter,
                     'txt': self.text_splitter}
        try:
            documents = loader.load()
        except (FileNotFoundError, ValueError, ImportError) as e:
            raise LoaderError('[Error]: Could not load dataset') from e
        splitter = splitters.get(ext)
        if splitter is None:
            return []
        return splitter(documents, chunk_size, overlap)

    def load(self, chunk_size: int, overlap: int, path: str = './data/raw'
             ) -> list[Document]:
        split_md = self.load_from_extension(chunk_size, overlap, 'md', path)
        split_py = self.load_from_extension(chunk_size, overlap, 'py', path)
        split_txt = self.load_from_extension(chunk_size, overlap, 'txt', path)
        return split_md + split_py + split_txt

    def python_splitter(self, documents: list[Document], chunk_size: int,
                        overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True)
        chunks = []
        for document in documents:
            document_chunks = splitter.split_documents([document])
            document_chunks = self._add_character_indices(document_chunks)
            chunks.extend(document_chunks)
        return chunks

    def text_splitter(self, documents: list[Document], chunk_size: int,
                      overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=overlap,
                                                  add_start_index=True)
        chunks = []
        for document in documents:
            document_chunks = splitter.split_documents([document])
            document_chunks = self._add_character_indices(document_chunks)
            chunks.extend(document_chunks)
        return chunks

    def markdown_splitter(self, documents: list[Document], chunk_size: int,
                          overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
            separators=[
                "\n### ",
                "\n## ",
                "\n# ",
                "\n\n",
                "\n",
                " ",
                "",])
        chunks = []
        for document in documents:
            document_chunks = splitter.split_documents([document])
            document_chunks = self._add_character_indices(document_chunks)
            chunks.extend(document_chunks)
        return chunks
'
BM25S :
'
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
    k: int = Field(default=5)

    @classmethod
    def index(cls, documents: list[Document], k: int = 4,
              path: str | None = None) -> "BM25SRetriever":
        corpus = [doc.page_content for doc in documents]
        tokenized = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        try:
            retriever.index(tokenized)
        except ValueError:
            raise RetrieverError('[ERROR]: Cannot index corpus')
        if path:
            retriever.save(path, corpus=corpus)
        return cls(retriever=retriever, documents=documents, k=k)

    @classmethod
    def from_index(cls, path, documents: list[Document], k: int = k
                   ) -> "BM25SRetriever":
        index = bm25s.BM25.load(path, load_corpus=False)
        return cls(retriever=index, documents=documents, k=k)

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
'
Chroma :
'
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
              embeddings: HuggingFaceEmbeddings, k: int = k,
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
                   k: int = k) -> "VectorRetriever":
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
'