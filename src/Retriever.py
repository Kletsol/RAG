import bm25s
from pydantic import Field
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter


class RetrieverError(Exception):
    pass


class LoaderSplitter:

    def load(self, chunk_size: int, overlap: int, ext: str, path: str = './data/raw') -> list[Document]:
        ext = ext.lstrip(".")
        loader = DirectoryLoader(path, glob=f"**/*.{ext}", loader_cls=TextLoader)
        splitters = {'md': self.markdown_splitter,
                     'py': self.python_splitter,
                     'txt': self.text_splitter}
        try:
            documents = loader.load()
        except (FileNotFoundError, ValueError, ImportError) as e:
            raise RetrieverError('[Error]: Could not load dataset') from e
        splitter = splitters.get(ext)
        if splitter is None:
            return []
        return splitter(documents, chunk_size, overlap)

    def python_splitter(self, documents: list[Document], chunk_size: int,
                        overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=overlap)
        return splitter.split_documents(documents)

    def text_splitter(self, documents: list[Document], chunk_size: int,
                      overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=overlap)
        return splitter.split_documents(documents)

    def markdown_splitter(self, documents: list[Document], chunk_size: int,
                          overlap: int) -> list[Document]:
        chunks = []
        headers = [("#", "Header 1"),
                   ("##", "Header 2"),
                   ("###", "Header 3")]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers, strip_headers=False)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap)
        for document in documents:
            header_docs = header_splitter.split_text(document.page_content)

            # Original metadata recovery
            for header_doc in header_docs:
                header_doc.metadata.update(document.metadata)
            doc_chunks = recursive_splitter.split_documents(header_docs)
            chunks.extend(doc_chunks)

        return chunks


class BM25SRetriever(BaseRetriever):

    bm25_index: bm25s.BM25 = Field(description="BM25S index")
    documents: list[Document] = Field(description="Langchain documents")
    k: int = Field(default=4, description="number of documents to be returned")
    corpus: list[str] | None = None

    @classmethod
    def index(cls, documents: list[Document], k: int = 4, path: str | None = None) -> "BM25SRetriever":
        corpus = [doc.page_content for doc in documents]
        tokenized = bm25s.tokenize(corpus)
        index = bm25s.BM25()
        index.index(tokenized)
        if path:
            index.save(path, corpus=corpus)
        return cls(bm25_index=index, documents=documents, k=k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        tokenized_query = bm25s.tokenize([query])
        n = min(self.k, len(self.documents))
        results, _ = self.bm25_index.retrieve(tokenized_query, k=n)
        documents = []
        for i in range(results.shape[1]):
            idx = int(results[0, i])
            documents.append(self.documents[idx])
        return documents

    def save(self, path: str = "./data/processed") -> None:
        try:
            self.bm25_index.save(path)
            with open(f"{path}/corpus.json", 'w') as f:
                json.dump(self.corpus, f)
            docs = [doc.model_dump() for doc in self.documents]
            with open(f"{path}/documents.json", 'w') as f:
                json.dump(docs, f)
        except PermissionError:
            raise RetrieverError("[ERROR]: Cannot save documents in folder -"
                                 " permission denied")

    # def load(self, path: str = "./data/processed") -> None:
