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

        # chunks = []
        # headers = [("#", "Header 1"),
        #            ("##", "Header 2"),
        #            ("###", "Header 3")]
        # header_splitter = MarkdownHeaderTextSplitter(
        #     headers_to_split_on=headers, strip_headers=False)
        # recursive_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size, chunk_overlap=overlap)
        # for document in documents:
        #     header_docs = header_splitter.split_text(document.page_content)

        #     # Original metadata recovery
        #     for header_doc in header_docs:
        #         header_doc.metadata.update(document.metadata)
        #     doc_chunks = recursive_splitter.split_documents(header_docs)
        #     chunks.extend(doc_chunks)

        # return chunks
