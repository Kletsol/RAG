from langchain_core.documents import Document


class RRF:

    @staticmethod
    def _source_key_from_document(document: Document) -> tuple[str, int, int]:
        """Returns a Document's source, first and last character
           index as a tuple
        """
        return (document.metadata["file_path"],
                document.metadata["first_character_index"],
                document.metadata["last_character_index"])

    def _rrf(self, bm25_documents: list[Document],
             chroma_documents: list[Document],
             k: int, rrf_k: int = 60) -> list[Document]:
        """A Reciprocal Rank Fusion algorithm used to merge results
           from both retrievers

        Args:
            bm25_documents (list[Document]): The results retrieved by BM25s
            chroma_documents (list[Document]): The results retrieved by Chroma
            k (int): The max number of sources to use
            rrf_k (int, optional): An empirical constant. Defaults to 60.

        Returns:
            list[Document]: The top-k documents sorted by rrf
        """
        scores: dict[tuple[str, int, int], float] = {}
        documents: dict[tuple[str, int, int], Document] = {}
        # ------
        # BM25 ranking
        # ------
        for rank, document in enumerate(bm25_documents[:k], start=1):
            key = self._source_key_from_document(document)
            scores[key] = (scores.get(key, 0.0) + 1.0 / (rrf_k + rank))
            documents[key] = document
        # ------
        # Chroma ranking
        # ------
        for rank, document in enumerate(chroma_documents[:k], start=1):
            key = self._source_key_from_document(document)
            scores[key] = (scores.get(key, 0.0) + 1.0 / (rrf_k + rank))
            documents[key] = document

        sorted_keys = sorted(scores.keys(),
                             key=lambda key: scores[key], reverse=True)

        return [documents[key] for key in sorted_keys[:k]]
