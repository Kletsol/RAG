import json
import os
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from .Fusion import RRF
from .LLM import LLM
from .LoaderSplitter import LoaderError, LoaderSplitter
from .Models import (
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
                 processed_directory: str = "./data/processed",
                 bonus: bool = False):
        self.raw_dir = Path(raw_directory)
        self.processed_dir = Path(processed_directory)
        self.bm25s_dir = (self.processed_dir / "bm25")
        self.vector_dir = (self.processed_dir / "vector")
        self.bm25_retriever = None
        self.vector_retriever = None
        self.embeddings: HuggingFaceEmbeddings | None = None
        self.vector: bool = bonus
        self.merger = None
        self.llm = LLM | None

    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 200:
            raise ProcessorError(
                "[ERROR]: max_chunk_size cannot be lower than 200")
        loader = LoaderSplitter()
        try:
            documents = loader.load(max_chunk_size, overlap=15,
                                    path=str(self.raw_dir))
        except LoaderError as e:
            raise ProcessorError(e)
        if not documents:
            raise ProcessorError("[ERROR]: No documents found")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        # ------
        # BM25
        # ------
        try:
            self.bm25_retriever = BM25SRetriever.index(
                documents=documents, k=5, path=str(self.bm25s_dir))
            if self.bm25_retriever is not None:
                self.bm25_retriever.save(str(self.bm25s_dir))
        except RetrieverError as e:
            raise ProcessorError("[ERROR]: Could not create BM25 index") from e
        # ------
        # Chroma
        # ------
        if self.vector is True:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vector_retriever = VectorRetriever.index(
                    documents=documents, embeddings=self.embeddings,
                    k=5, path=str(self.vector_dir))
            except Exception as e:
                raise ProcessorError(
                    "[ERROR]: Could not create Chroma index") from e

    def load(self, k: int = 10) -> None:
        documents_path = (self.bm25s_dir / "documents.json")
        if not documents_path.exists():
            raise ProcessorError("[ERROR]: No index found")
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
        if self.vector is True and self.embeddings is None and \
                self.vector_retriever is None:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vector_retriever = (VectorRetriever.from_index(
                    path=str(self.vector_dir),
                    embeddings=self.embeddings, k=k))
            except Exception:
                raise ProcessorError("[ERROR]: Could not load Chroma index")

    @staticmethod
    def _document_to_source(document: Document) -> MinimalSource:
        return MinimalSource(file_path=document.metadata["file_path"],
                             first_character_index=document.metadata[
                                 "first_character_index"],
                             last_character_index=document.metadata[
                                 "last_character_index"])

    def search(self, query: str, k: int = 5) -> MinimalSearchResults:
        self.load(k=k)
        if not query.strip():
            raise ProcessorError("[ERROR]: Empty query")
        if k <= 0:
            raise ProcessorError("[ERROR]: k must be greater than 0")
        if (self.bm25_retriever is None) or (
                self.vector is True and self.vector_retriever is None):
            raise ProcessorError("[ERROR]: No index loaded")

        try:
            bm25_documents = (self.bm25_retriever.invoke(query))
            if self.vector is True:
                chroma_documents = (self.vector_retriever.invoke(query))
        except Exception as e:
            raise ProcessorError("[ERROR]: Retrieval failed") from e

        if self.vector is True:
            documents = self.merger._rrf(bm25_documents=bm25_documents,
                                         chroma_documents=chroma_documents,
                                         k=k)
        else:
            documents = bm25_documents
        sources = [self._document_to_source(doc) for doc in documents]

        return MinimalSearchResults(
            question_id=str(uuid.uuid4()), question=query,
            retrieved_sources=sources)

    def search_dataset(self, dataset_path: str, k: int, save_directory: str,
                       ) -> StudentSearchResults:
        if k <= 0:
            raise ProcessorError("[ERROR]: k must be greater than 0")
        if self.vector is True:
            self.merger = RRF()
        # self.load(k=k)
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = RagDataset.model_validate(json.load(f))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise ProcessorError("\033[1;31m[ERROR]: Could not load dataset: "
                                 f"\033[0;0m {e}")

        results = []

        length = len(dataset.rag_questions)
        for question in tqdm(dataset.rag_questions,
                             desc=f'Processing {length} questions',
                             colour='yellow'):
            result = self.search(query=question.question, k=k)
            result.question_id = question.question_id
            results.append(result)
        student_results = StudentSearchResults(search_results=results, k=k)
        # ------
        # Save
        # ------
        file_basename = os.path.basename(dataset_path)
        os.makedirs(save_directory, exist_ok=True)
        output_path = (f"{save_directory}/{file_basename}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(student_results.model_dump(), f,
                          ensure_ascii=False, indent=2)
        except OSError:
            raise ProcessorError("[ERROR]: Cannot save search results")
        return student_results

    @staticmethod
    def _load_source_content(source: MinimalSource) -> str:
        path = Path(source.file_path)
        if not path.exists():
            raise ProcessorError(f"[ERROR]: Source not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            raise ProcessorError(
                f"[ERROR]: Could not read source: {path}")
        return content[
            source.first_character_index: source.last_character_index + 1]

    def _build_context(self, sources: list[MinimalSource]) -> str:
        contexts = []
        for index, source in enumerate(sources, start=1):
            content = self._load_source_content(source)
            contexts.append(f"[Source {index}]\n"
                            f"File: {source.file_path}\n"
                            f"{content}")
        return "\n\n".join(contexts)

    def answer(self, query: str, k: int = 5) -> str:
        if self.llm is None:
            self.llm = LLM()
        search_result = self.search(query=query, k=k)
        context = self._build_context(search_result.retrieved_sources)
        return str(self.llm._generate_answer(question=query, context=context))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> str:
        # ------
        # Read results file
        # ------
        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                student_results = (
                    StudentSearchResults.model_validate(json.load(f)))
        except OSError:
            raise ProcessorError(
                "[ERROR]: File not found - "
                f"{os.path.basename(student_search_results_path)}")
        except json.JSONDecodeError:
            raise ProcessorError("[ERROR]: Invalid json")
        except ValueError:
            raise ProcessorError("[ERROR]: Could not load search results")
        # ------
        # Answer
        # ------
        answers = []

        if self.llm is None:
            self.llm = LLM()

        for result in tqdm(student_results.search_results, desc='Answering',
                           colour='green'):
            context = self._build_context(result.retrieved_sources)
            response = self.llm._generate_answer(question=result.question,
                                                 context=context)
            answers.append(
                MinimalAnswer(question_id=result.question_id,
                              question=result.question,
                              retrieved_sources=result.retrieved_sources,
                              answer=response))
        final_results = (StudentSearchResultsAndAnswer(
            search_results=answers, k=student_results.k))
        # ------
        # Save
        # ------
        file_basename = os.path.basename(student_search_results_path)
        os.makedirs(save_directory, exist_ok=True)
        output_path = (f"{save_directory}/{file_basename}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_results.model_dump(), f, ensure_ascii=False,
                          indent=2)
        except OSError as e:
            raise ProcessorError("[ERROR]: Could not save answers") from e
        return str(output_path)
