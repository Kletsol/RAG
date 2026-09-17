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
        """Takes a list of Documents and adds its source,
        first and last character index

        Args:
            chunks (list[Document]): a file splitted in Documents

        Returns:
            list[Document]: the augmented list of Documents
        """
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content)
            chunk.metadata["file_path"] = chunk.metadata["source"]
            chunk.metadata["first_character_index"] = start
            chunk.metadata["last_character_index"] = end - 1
        return chunks

    def load_from_extension(self, chunk_size: int, overlap: int, ext: str,
                            path: str = './data/raw') -> list[Document]:
        """Loads files depending on the given extension
        and returns a splitted version of it

        Args:
            chunk_size (int): The size of the chunks resulting from
                              the splitting process
            overlap (int): The overlap needed between two chunks
            ext (str): The extension we want to get
            path (str, optional): The path of the data to process.
                                  Defaults to './data/raw'.

        Returns:
            list[Document]: All the splitted files with the given extension
        """
        loader = DirectoryLoader(path, glob=f"**/*.{ext}",
                                 loader_cls=TextLoader)
        splitters = {'py': self.python_splitter,
                     'md': self.markdown_splitter,
                     'txt': self.markdown_splitter}
        try:
            documents = loader.load()
        except (FileNotFoundError, ValueError, ImportError) as e:
            raise LoaderError(f"[Error]: Could not load dataset: {e}")
        except RuntimeError as e:
            raise LoaderError(f"[ERROR]: {e} - Permission denied")
        splitter = splitters.get(ext)
        if splitter is None:
            return []
        return splitter(documents, chunk_size, overlap)

    def load(self, chunk_size: int, overlap: int, path: str = './data/raw'
             ) -> list[Document]:
        """An index loader: gets the files we want, splits it and returns
        the whole index
        Args:
            chunk_size (int): The size of the chunks resulting from
                              the splitting process
            overlap (int): The overlap needed between two chunks
            path (str, optional): The path of the data to process.
                                  Defaults to './data/raw'.

        Returns:
            list[Document]: An index built from multiple files
        """
        split_md = self.load_from_extension(chunk_size, overlap, 'md', path)
        split_py = self.load_from_extension(chunk_size, overlap, 'py', path)
        split_txt = self.load_from_extension(chunk_size, overlap, 'txt', path)
        return split_md + split_txt + split_py

    def python_splitter(self, documents: list[Document], chunk_size: int,
                        overlap: int) -> list[Document]:
        """A splitter used for python (.py) files"""
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

    def markdown_splitter(self, documents: list[Document], chunk_size: int,
                          overlap: int) -> list[Document]:
        """A splitter used for markdown and text (.md | .txt) files"""
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
