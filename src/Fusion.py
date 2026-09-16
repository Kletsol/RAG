from langchain_core.documents import Document


class RRF:

    @staticmethod
    def _source_key_from_document(document: Document) -> tuple[str, int, int]:
        return (document.metadata["file_path"],
                document.metadata["first_character_index"],
                document.metadata["last_character_index"])

    def _rrf(self, bm25_documents: list[Document],
             chroma_documents: list[Document],
             k: int, rrf_k: int = 60) -> list[Document]:
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
