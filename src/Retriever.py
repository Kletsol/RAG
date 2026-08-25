import bm25s
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter


class RetrieverError(Exception):
    pass


class LoaderSplitter:

    def load(self, chunk_size: int, overlap: int, ext: str, language: Language, path: str = './data/raw') -> list[Document]:
        ext = ext.lstrip(".")
        loader = DirectoryLoader(path, glob=f"**/*.{ext}", loader_cls=TextLoader)
        splitters = {Language.MARKDOWN: self.markdown_splitter,
                     Language.PYTHON: self.python_splitter,
                     Language.TEXT: self.text_splitter}
        try:
            documents = loader.load()
        except (FileNotFoundError, ValueError, ImportError) as e:
            raise RetrieverError('[Error]: Could not load dataset') from e
        splitter = splitters.get(language)
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

    def index():
        pass
