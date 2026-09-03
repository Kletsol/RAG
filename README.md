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





The pipeline is driven by four commands, in order: index the corpus, search a whole
dataset, score the results with the moulinette, then generate answers. The search and
answer single-query commands shown earlier behave the same way on one question at a
time.

1. Index the corpus once:
uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indices saved under data/processed/

2. Search a dataset. Always scope --save_directory by dataset (UnansweredQuestions
or AnsweredQuestions): the public datasets share file names, so writing every run into
the same folder would overwrite previous results.
uv run python -m src search_dataset
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
--k 10
--save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

3. Score with the moulinette (rename moulinette-ubuntu/-fedora to moulinette
first). The student results come first, the ground-truth AnsweredQuestions dataset
second:
./moulinette evaluate_student_search_results
data/output/search_results/UnansweredQuestions/dataset_docs_public.json
data/datasets/AnsweredQuestions/dataset_docs_public.json
--k 10 --max_context_length 2000
Student data is valid: True
Evaluation Results
Recall@1: 0.450 Recall@3: 0.590 Recall@5: 0.650 Recall@10: 0.720

4. Generate answers from the search results:
uv run python -m src answer_dataset
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
--save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions ... Processed 100 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json


======================================================


Je suis en train d'ecrire un RAG hybride pour un projet encadre. J'ai deja mon splitter, mes deux retrievers (BM25s et Chroma) et, en principe, un controller (Processor) qui me permet d'effectuer les actions necessaires :
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
            chunk.metadata["file_path"] = chunk.metadata["source"]
            chunk.metadata["first_character_index"] = start
            chunk.metadata["last_character_index"] = end - 1
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
BM25S:
'
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
            with open(directory / "documents.json", 'w',
                      encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
        except PermissionError as e:
            raise RetrieverError("[ERROR]: Cannot save documents in folder -"
                                 " permission denied") from e
'
Chroma:
'
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

        first_batch = documents[:250]

        vectorstore = Chroma.from_documents(
            documents=first_batch, embedding=embeddings,
            persist_directory=path, collection_name="test")

        remaining_docs = documents[250:]
        if remaining_docs:
            for i in tqdm(range(0, len(remaining_docs), 250)):
                batch = remaining_docs[i: i + 250]
                vectorstore.add_documents(batch)

        return cls(vectorstore=vectorstore, k=k, embeddings=embeddings)

    @classmethod
    def from_index(cls, path: str,
                   embeddings: HuggingFaceEmbeddings | None,
                   k: int = 5) -> "VectorRetriever":
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
Processor:
'
import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import chat

from .LoaderSplitter import LoaderSplitter
from .Models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .retrievers.BM25S import BM25SRetriever, RetrieverError
from .retrievers.Chroma import VectorRetriever


class ProcessorError(Exception):
    pass


class Processor:

    def __init__(self, raw_directory: str = "./data/raw",
                 processed_directory: str = "./data/processed"):
        self.raw_dir = Path(raw_directory)
        self.processed_dir = Path(processed_directory)
        self.bm25s_dir = (self.processed_dir / "bm25")
        self.vector_dir = (self.processed_dir / "vector")
        self.bm25_retriever = None
        self.vector_retriever = None
        self.embeddings = None

    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 200:
            raise ProcessorError(
                '[ERROR]: max_chunk_size cannot be lower than 200')
        loader = LoaderSplitter()
        try:
            documents = loader.load(max_chunk_size, overlap=50,
                                    path=str(self.raw_dir))
        except Exception as e:
            raise ProcessorError("[ERROR]: Could not load dataset") from e
        if not documents:
            raise ProcessorError("[ERROR]: No documents found")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # ------
        # BM25
        # ------
        try:
            self.bm25_retriever = BM25SRetriever.index(
                documents=documents, k=5, path=str(self.bm25s_dir))
        except RetrieverError as e:
            raise ProcessorError("[ERROR]: Could not create BM25 index") from e
        # ------
        # Chroma
        # ------
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_retriever = VectorRetriever.index(
                documents=documents, embeddings=self.embeddings,
                k=5, path=str(self.vector_dir))
        except Exception as e:
            raise ProcessorError(
                "[ERROR]: Could not create Chroma index") from e

    def load(self, k: int = 5) -> None:
        documents_path = (self.bm25s_dir / "documents.json")
        if not documents_path.exists():
            raise ProcessorError("[ERROR]: documents.json not found")
        try:
            with open(documents_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            documents = [Document(
                page_content=item["page_content"],
                metadata=item["metadata"])for item in data]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise ProcessorError("[ERROR]: Could not load documents") from e
        try:
            self.bm25_retriever = (BM25SRetriever.from_index(
                path=str(self.bm25s_dir), documents=documents, k=k))
        except RetrieverError as e:
            raise ProcessorError("[ERROR]: Could not load BM25S index") from e
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_retriever = (VectorRetriever.from_index(
                path=str(self.vector_dir), embeddings=self.embeddings, k=k))
        except Exception as e:
            raise ProcessorError("[ERROR]: Could not load Chroma index") from e

    @staticmethod
    def _source_key_from_document(document: Document) -> tuple:
        return (document.metadata["file_path"],
                document.metadata["first_character_index"],
                document.metadata["last_character_index"])

    @staticmethod
    def _source_key(source: MinimalSource) -> tuple:
        return (source.file_path,
                source.first_character_index,
                source.last_character_index)

    @staticmethod
    def _document_to_source(document: Document) -> MinimalSource:
        return MinimalSource(file_path=document.metadata["file_path"],
                             first_character_index=document.metadata[
                                 "first_character_index"],
                             last_character_index=document.metadata[
                                 "last_character_index"])

    def _rrf(bm25_documents: list[Document], chroma_documents: list[Document],
             k: int, rrf_k: int = 60) -> list[Document]:
        scores: dict[tuple, float] = {}
        documents: dict[tuple, Document] = {}
        # ------
        # BM25 ranking
        # ------
        for rank, document in enumerate(bm25_documents[:k], start=1):
            key = Processor._source_key_from_document(document)
            scores[key] = (scores.get(key, 0.0) + 1.0 / (rrf_k + rank))
            documents[key] = document
        # ------
        # Chroma ranking
        # ------
        for rank, document in enumerate(chroma_documents[:k], start=1):
            key = Processor._source_key_from_document(document)
            scores[key] = (scores.get(key, 0.0) + 1.0 / (rrf_k + rank))
            documents[key] = document

        sorted_keys = sorted(scores.keys(),
                             key=lambda key: scores[key], reverse=True)

        return [documents[key] for key in sorted_keys[:k]]

    def search(self, query: str, k: int = 5) -> MinimalSearchResults:
        if not query.strip():
            raise ProcessorError("[ERROR]: Empty query")
        if k <= 0:
            raise ProcessorError("[ERROR]: k must be greater than 0")
        if (self.bm25_retriever is None or self.vector_retriever is None):
            raise ProcessorError("[ERROR]: No index loaded")
        try:
            bm25_documents = (self.bm25_retriever.invoke(query))
            chroma_documents = (self.vector_retriever.invoke(query))
        except Exception as e:
            raise ProcessorError("[ERROR]: Retrieval failed") from e
        documents = self._rrf(bm25_documents=bm25_documents,
                              chroma_documents=chroma_documents, k=k)
        sources = [self._document_to_source(doc) for doc in documents]
        return MinimalSearchResults(
            question_id=str(uuid.uuid4()), question=query,
            retrieved_sources=sources)

    def search_dataset(self, dataset_path: str, k: int, save_directory: str
                       ) -> StudentSearchResults:
        if k <= 0:
            raise ProcessorError("[ERROR]: k must be greater than 0")
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise ProcessorError("[ERROR]: Could not load dataset") from e

        results = []

        for question in dataset.rag_questions:
            result = self.search(query=question.question, k=k)
            result.question_id = question.question_id
            results.append(result)

        student_results = StudentSearchResults(search_results=results, k=k)
        output_directory = Path(save_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = (output_directory / "search_results.json")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(student_results.model_dump(), f,
                          ensure_ascii=False, indent=2)
        except OSError as e:
            raise ProcessorError("[ERROR]: Cannot save search results") from e
        return student_results

    @staticmethod
    def _load_source_content(source: MinimalSource) -> str:
        path = Path(source.file_path)
        if not path.exists():
            raise ProcessorError(f"[ERROR]: Source not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ProcessorError(
                f"[ERROR]: Could not read source: {path}") from e
        return content[
            source.first_character_index: source.last_character_index]

    def _build_context(self, sources: list[MinimalSource]) -> str:
        contexts = []
        for index, source in enumerate(sources, start=1):
            content = self._load_source_content(source)
            contexts.append(f"[Source {index}]\n"
                            f"File: {source.file_path}\n"
                            f"{content}")
        return "\n\n".join(contexts)

    def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
You are a retrieval-augmented generation assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use external knowledge.
- Do not invent information.
- If the context does not contain enough information,
  say that the answer cannot be determined from the provided context.
- Be concise and directly answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
        try:
            response = chat(
                model="qwen3:0.6b",
                messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            raise ProcessorError("[ERROR]: LLM generation failed") from e
        return response.message.content.strip()

    def answer(self, query: str, k: int = 5) -> str:
        search_result = self.search(query=query, k=k)
        context = self._build_context(search_result.retrieved_sources)
        return self._generate_answer(question=query, context=context)

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> str:
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_results = (
                    StudentSearchResults.model_validate(json.load(f)))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise ProcessorError(
                "[ERROR]: Could not load search results") from e

        answers = []

        for result in student_results.search_results:
            context = self._build_context(result.retrieved_sources)
            response = self._generate_answer(question=result.question,
                                             context=context)
            answers.append(
                MinimalAnswer(question_id=result.question_id,
                              question=result.question,
                              retrieved_sources=result.retrieved_sources,
                              answer=response))
        final_results = (StudentSearchResultsAndAnswer(
            search_results=answers, k=student_results.k))
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (output_dir / "answers.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_results.model_dump(), f, ensure_ascii=False,
                          indent=2)
        except OSError as e:
            raise ProcessorError("[ERROR]: Could not save answers") from e
        return str(output_path)

    def evaluate(self, student_search_results_path: str, dataset_path: str
                 ) -> None:
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_results = (
                    StudentSearchResults.model_validate(json.load(f)))
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise ProcessorError(
                "[ERROR]: Could not load evaluation data") from e

        ground_truth = {}

        for question in dataset.rag_questions:
            if isinstance(question, AnsweredQuestion):
                ground_truth[question.question_id] = question.sources
        recalls = []
        for result in student_results.search_results:
            expected_sources = ground_truth.get(result.question_id)
            if not expected_sources:
                continue
            expected = {self._source_key(src) for src in expected_sources}
            retrieved = {self._source_key(src)
                         for src in result.retrieved_sources}
            hits = expected & retrieved
            recall = len(hits) / len(expected)
            recalls.append(recall)
            print(f"{result.question_id}: "
                  f"recall@{student_results.k} = {recall:.4f}")
        if not recalls:
            print("No questions available for evaluation")
            return
        mean_recall = sum(recalls) / len(recalls)
        print(f"\nMean recall@{student_results.k}: {mean_recall:.4f}")
'
Mon probleme, a present, est que je dois etre en mesure d'executer chaque etape via CLI, donc en reexecutant le programme a chaque fois, chose pour laquelle mon Processor n'a pas ete concu. Les etapes sont les suivantes :
'
The pipeline is driven by four commands, in order: index the corpus, search a whole
dataset, score the results with the moulinette, then generate answers. The search and
answer single-query commands shown earlier behave the same way on one question at a
time.

1. Index the corpus once:
uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indices saved under data/processed/

2. Search a dataset. Always scope --save_directory by dataset (UnansweredQuestions
or AnsweredQuestions): the public datasets share file names, so writing every run into
the same folder would overwrite previous results.
uv run python -m src search_dataset
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
--k 10
--save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

3. Score with the moulinette (rename moulinette-ubuntu/-fedora to moulinette
first). The student results come first, the ground-truth AnsweredQuestions dataset
second:
./moulinette evaluate_student_search_results
data/output/search_results/UnansweredQuestions/dataset_docs_public.json
data/datasets/AnsweredQuestions/dataset_docs_public.json
--k 10 --max_context_length 2000
Student data is valid: True
Evaluation Results
Recall@1: 0.450 Recall@3: 0.590 Recall@5: 0.650 Recall@10: 0.720

4. Generate answers from the search results:
uv run python -m src answer_dataset
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
--save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions ... Processed 100 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json
'