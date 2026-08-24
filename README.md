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




Je travaille sur un projet dont le but global est de creer un RAG, avec plusieurs contraintes. Le sujet est le suivant :

You will build a Retrieval-Augmented Generation system that answers questions
about a codebase. You ingest the provided vLLM repository into a searchable index,
retrieve the most relevant snippets for a question, generate an answer from them with
Qwen/Qwen3-0.6B, and measure retrieval quality with recall@k. Your system is judged on
whether it retrieves the right source locations and produces answers grounded in them.

1. Indexing the codebase

Everything starts with the index. Read the files you judge useful from the vLLM repos-
itory shipped in the attachments, split each one into chunks, and persist an index that
retrieval can query in milliseconds. Indexing the whole corpus must take at most 5
minutes.
A Python file and a Markdown page do not break apart the same way, so your program
must implement two distinct chunking strategies:
• Python code chunking,
• Markdown / text chunking.

For retrieval itself, implement at least one of the two classic lexical methods. The choice
is yours:
• TF-IDF,
• BM25

We'll use a BM25s retriever.

2. Retrieval

With the index built, you can search it. Given a question, your system returns the
top-k most relevant snippets. Each result is a source location: a file_path and the
character range (first_character_index, last_character_index) it covers, at most
2000 characters wide.
Retrieval must work for a single query and in batch over a whole dataset of questions read
from JSON. On the reference datasets, your system must reach at least 80 % recall@5 on
docs questions and 50 % on code questions.

3. Answer generation

With the right snippets retrieved, the system generates a natural-language answer using
Qwen/Qwen3-0.6B. Pass the retrieved context to the model within its token budget, and
produce structured JSON following the provided pydantic models.
A satisfactory answer is:
• Coherent and understandable,
• Grounded in the retrieved sources, with no major hallucination,
• On point: it answers the question actually asked.

4. Data Models

The following pydantic models are mandatory :

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

5. Output

Each command writes a JSON file that conforms to the provided pydantic models:
• For search operations: Use StudentSearchResults model with:
◦ search_results: List of MinimalSearchResults containing question_id, question and retrieved_sources
◦ k: Number of results requested
• For answer generation: Use StudentSearchResultsAndAnswer model with:
◦ search_results: List of MinimalAnswer containing question_id, question, retrieved_sources, and answer
◦ k: Number of results requested
• Source information: Each MinimalSource contains:
◦ file_path: path to the source file, relative to your project root and written exactly as in the ingested corpus (e.g. data/raw/vllm-0.10.1/...); it is compared verbatim to the reference
◦ first_character_index: Starting character position
◦ last_character_index: Ending character position

6. Running the full pipeline

The pipeline is driven by four commands, in order: index the corpus, search a whole
dataset, score the results with the moulinette, then generate answers. The search and
answer single-query commands shown earlier behave the same way on one question at a
time.
1- Index the corpus once:
uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indices saved under data/processed/
2- Search a dataset. Always scope --save_directory by dataset (UnansweredQuestions
or AnsweredQuestions): the public datasets share file names, so writing every run into
the same folder would overwrite previous results.
uv run python -m src search_dataset
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
--k 10
--save_directory data/output/search_results/UnansweredQuestions
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
3- Score with the moulinette (rename moulinette-ubuntu/-fedora to moulinette
first). The student results come first, the ground-truth AnsweredQuestions dataset
second.
4- Generate answers from the search results:
uv run python -m src answer_dataset
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
--save_directory data/output/search_results_and_answer/UnansweredQuestions
Loaded 100 questions ... Processed 100 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json

7. Evaluation

For each question, recall@k is the share of its correct sources that you retrieve in your
top-k results. A correct source counts as found when one of your results is in the same
file and overlaps its character range.
The overlap bar is low (an IoU of 0.05), so you do not need to match the reference span
exactly: retrieving a chunk that covers the right region of the right file is enough. A
result in a different file never counts, which is why file_path must be exact.
Your system must respect some minimal performances, listed below:
• Indexing time: at most 5 minutes for the whole corpus.
• Retrieval throughput: at most 90 seconds for 200 questions.
• Recall@5: at least 80% on docs questions and 50% on code questions.

Je me suis deja renseigne sur plusieurs outils utiles, et pense utiliser, entre autres:
- BM25s pour le retriever,
- langchain pour ses document loaders / text splitters
- transformers pour acceder a Qwen
Toutes ces librairies etant deja installees dans mon environnement virtuel (voir pyproject.toml):
'
[project]
name = "rag"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "bm25s>=0.3.10",
    "fire>=0.7.1",
    "pydantic>=2.13.4",
    "transformers>=5.15.1",
]
'
Partant de la, quelles sont les etapes a suivre pour commencer ?