*This project has been created as part of the 42 curriculum by lbonnet*

# RAG against the machine 💭

<span style="color:turquoise">

## 📝 Description
</span>

This project's goal is to build a Retrieval-Augmented Generation system (RAG) that answers questions about a codebase. To achive that, it has to ingest a provided repository into a searchable index, retrieve the most relevant snippets for a question, generate an answer from them and measure retrieval quality with recall@k.

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

- https://github.com/KeroBeros68/Obsidian-vault/tree/main/ia :  

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

During the execution of this project, I faced two main challenges :
- First, RAG is a project that introduces us to many new concepts. Consequently, this requires reading and assimilating a large amount of documentation in a short time which can be overwhelming.
- The subject states that indexing time must take at most 5 minutes for the whole corpus. However, adding a vector index makes this limit impossible to meet. I spent a lot of time trying to reduce the indexing time before finally asking one of my colleagues, who told me that this limit only applies to the lexical index.

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
