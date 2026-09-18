*This project has been created as part of the 42 curriculum by lbonnet*

# RAG against the machine 💭

<span style="color:turquoise">

## 📝 Description
</span>

This project's goal is to build a Retrieval-Augmented Generation system (RAG) that answers  
questions about a codebase.

To achieve that, it has to ingest a provided repository into a searchable index, retrieve the  
most relevant snippets for a question, generate an answer from them and measure retrieval  
quality with recall@k.

For more details on how it works, please read the [System architecture] section.

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

You will find usage examples in the section of the same name at the end of this README.


<span style="color:turquoise">

## 📚 Resources
</span>

**Links:**

Some articles, references and tutorials I used during the elaboration of this project:

- https://realpython.com/llamaindex-examples/

- https://github.com/KeroBeros68/Obsidian-vault/tree/main/ia

- https://docs.langchain.com/oss/python/integrations/splitters

- https://docs.langchain.com/oss/python/integrations/embeddings

**AI usage:**

- AI was primarily used to gather and connect information, as RAG is a project  
that introduces many new concepts and modules that need to be linked together.

<span style="color:turquoise">

## 🚀 Additional sections
</span>

<span style="color:orange">

### -> System architecture
</span>

My RAG is built on the model of an Hybrid RAG and therefore follows these steps:

**1- Indexing:**

- Python (.py), text (.txt) and Markdown (.md) files are extracted from the corpus  
and splitted in chunks of size <max_chunk_size>
- Each line resulting from this operation is indexed using two strategies : lexical  
indexing and vectorial indexing

**2- Search:**

- Each query is treated both by two retrievers: BM25S (lexical) and Chroma (semantic)
- The results of each retriever are then fused using a Reciprocal Rank Fusion algorithm,  
(RRF for short) to get the most relevant sources

**3- Answer:**

- The sources are sent to an LLM as context along with the query, in a precise prompt
- The LLM produces an answer to the query, based only on the given context and not  
on its external knowledge.

<span style="color:orange">

### -> Chunking strategy
</span>

The size of each chunk is limited by <max_chunk_size>.

Depending on the type of file (.py, .md or .txt), as they have fundamentally different formats,  
the method RecursiveCharacterTextSplitter() (the splitter from Langchain I used) takes different  
parameters : that's why I designed two methods, one for python and one for markdown and text.

<span style="color:orange">

### -> Retrieval Method
</span>

As described in [System architecture], my project makes use of two retrievers:

- One lexical, with a BM25S algorithm
- One semantic, using Chroma and embeddings

The retrieved sources are then fused using an RRF to obtain their final ranks.

To better understand the RRF, here is an example:

    Vector results : [Doc A (rank 1), Doc C (rank 2), Doc B (rank 3)]
    BM25s Results  : [Doc B (rank 1), Doc A (rank 2), Doc D (rank 3)]

    RRF score: (Doc A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
    RRF score: (Doc B) = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
    RRF score: (Doc C) = 1/(60+2) + 0        = 0.0161

    Merged result : [Doc A, Doc B, Doc C, Doc D]

💡 In that example, 60 is an empirical constant that rewards a document for being well-ranked  
across multiple retrievers, rather than placing excessive importance on the top position.


<span style="color:orange">

### -> Performance analysis
</span>

||Naive RAG (BM25S only)|Hybrid RAG (BM25S + Chroma)|
|---|---|---|
|**Indexation time:**| ~30s for max_chunk_size=2000| ~4m30 for max_chunk_size=2000|
|**Recall on code:** |Recall@1: ~0.303|Recall@1: ~0.303|
||Recall@3: ~0.475|Recall@3: ~0.475|
||Recall@5: ~0.556|Recall@5: ~0.556|
||Recall@10: ~0.586|Recall@10: ~0.677|
|**Recall on docs:** |Recall@1: ~0.630|Recall@1: ~0.540|
||Recall@3: ~0.800|Recall@3: ~0.810|
||Recall@5: ~0.820|Recall@5: ~0.830|
||Recall@10: ~0.860|Recall@10: ~0.860|

As this table shows, the indexing step with Chroma is very time-consuming. This isn't really a  
problem, since it is a bonus, but it is worth being aware of for future projects.

<span style="color:orange">

### -> Design Decisions
</span>

My design was primarily dictated by the constraints of the subject :

- The repository's layout respects the structure given by the subject
- Same for the CLI
- Regarding the project itself, I chose BM25 over TF-IDF because BM25 incorporates a crucial  
adjustment for document length : this prevents very long documents from being unfairly  
ranked due to an excessively high word count.  
More generally, it's considered a more advanced model.
- The rest of the project is built around this idea, with semantic embeddings and hybrid  
retrieval as a bonus.

<span style="color:orange">

### -> Challenges faced
</span>

During the execution of this project, I faced two main challenges :
- First, RAG is a project that introduces us to many new concepts. Consequently, this requires  
reading and assimilating a large amount of documentation in a short time, which can be  
overwhelming.
- The subject states that indexing time must take at most 5 minutes for the whole corpus.  
However, adding a vector index makes this limit impossible to meet. I spent a lot of time  
trying to reduce the indexing time before finally asking one of my colleagues, who told  
me that this limit only applies to the lexical index.

<span style="color:orange">

### -> Example usage
</span>

To go through the whole pipeline, use the following steps :

-> Install the project:

    make install

-> Index the corpus:

    uv run python3 -m src index --max_chunk_size <int>

-> Search a dataset:

    uv run python3 -m src search_dataset --dataset_path <path> --k <int> --save_directory <dir>

-> Score your results:

    uv run python3 -m src evaluate --student_search_result_path <path> --dataset_path <path>

-> Generate answers from the search results:

    uv run python3 -m src answer_dataset --student_search_results_path <path> --save_directory <dir>

You can also search and answer a single query, using:

    uv run python3 -m src search --query <str> --k <int>

    uv run python3 -m src answer --query <str> --k <int>
