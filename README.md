# FilePilot

Basic desktop RAG app built with PyQt5, LlamaIndex, and Ollama.

## What it does

- Small always-on-top desktop window
- Load a local file or folder
- Build a local persisted RAG index with LlamaIndex
- Ask questions against the indexed content
- Use Ollama locally for both the LLM and embeddings

## Basic setup

1. Install Ollama and start it.
2. Pull the default models:
   - `ollama pull llama3.2`
   - `ollama pull nomic-embed-text`
3. Install Python dependencies:
   - `pip install -r requirements.txt`
4. Run the app:
   - `python app.py`

## Tests

1. Install test dependencies:
   - `pip install -r requirements-dev.txt`
2. Run the pytest suite:
   - `pytest`
3. Run the live DeepEval RAG evaluation:
   - `$env:RUN_DEEPEVAL_RAG="1"`
   - `pytest tests/test_rag_deepeval.py -m deepeval`
4. Generate an eval dataset automatically with RAGAS `TestsetGenerator`:
   - `python eval_file.py generate-dataset --output eval_dataset.json --testset-size 5 --rebuild-index`
5. Generate a dataset and run the eval in one command:
   - `python eval_file.py generate-and-evaluate --output eval_dataset.json --testset-size 5 --rebuild-index`

The DeepEval test uses the local RAG pipeline and a small fixture document in `tests/fixtures/rag_eval_source.txt`.

## Central config

All runtime settings now live in `config.json` at the project root.

- `source_path`
- `llm_model`
- `embedding_model`
- `ollama_base_url`
- chunking and retrieval settings
- basic window settings

The UI reads from that file and writes back to it. There is no separate model state source anymore.

## Notes

- The first version supports common text-like files through `SimpleDirectoryReader`.
- PDFs require `pypdf`.
- The app stores its index and state under `data/`.
- Rebuild the index if you change the embedding model.
