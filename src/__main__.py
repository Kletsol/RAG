import json
import time

import fire
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm
from transformers import pipeline

from .Models import StudentSearchResults
from .Processor import Processor, ProcessorError
from .Retriever import BM25SRetriever, LoaderSplitter, RetrieverError


class CLI:

    @staticmethod
    def index(max_chunk_size: int = 2000) -> None:
        processor = Processor()
        try:
            processor.index(max_chunk_size)
        except RetrieverError as e:
            raise RetrieverError(e)

    @staticmethod
    def search(query: str, k: int = 5) -> None:
        for i in tqdm(range(1000), desc="Ceci est un loooong test"):
            time.sleep(0.01)

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
        pass

    @staticmethod
    def answer(query: str, k: int = 5) -> None:
        pass

    @staticmethod
    def answer_dataset(student_search_results_path: str,
                       save_directory: str) -> None:
        pass

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        pass


# def main():
#     loader = LoaderSplitter()
#     chunks = loader.load(chunk_size=512, overlap=50, ext='.txt')
#     bm25sretriever = BM25SRetriever.index(chunks, k=4)
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#     vectorstore = Chroma.from_documents(chunks, embeddings)
#     vectorial_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     hybrid_retriever = EnsembleRetriever(retrievers=[bm25sretriever, vectorial_retriever], weights=[0.4, 0.6])

#     llm = pipeline("text-generation", model='Qwen/Qwen3-0.6B', device_map='auto')
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """Tu es un assistant expert et factuel.
#     Règles :
#     - Réponds UNIQUEMENT depuis le contexte fourni
#     - Si l'information est absente, dis-le clairement
#     - Cite toujours la source entre crochets [source]

#     Contexte :
#     {contexte}"""),
#         ("human", "{question}")
#     ])

#     chain_hybrid = (
#         {"contexte": hybrid_retriever, "question": RunnablePassthrough()}
#         | prompt | llm | StrOutputParser())

#     try:
#         with open('/home/lbonnet/Documents/Cursus/Github/RAG/data/datasets/UnansweredQuestions/dataset_code_public.json', 'r') as f:
#             data = json.load(f)
#         questions = data
#     except Exception as e:
#         raise ProcessorError(f"[ERROR]: {e}")

#     for q in questions['rag_questions']:
#         print(f"\n❓ {q['question']}")
#         print(f"💬 {chain_hybrid.invoke(q['question'])}")


if __name__ == "__main__":
    # try:
    # main()
    # except Exception as e:
    #     print(e)
    try:
        fire.Fire(CLI)
    except KeyboardInterrupt:
        print('\033[H\033[J')
        print("\033[0;32mAborted - See you soon :D\033[0;0m")
    except RetrieverError:
        pass
