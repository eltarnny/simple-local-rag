# Full local open source lightweight simple RAG

No usage of platforms like Ollama or LM Studio, or abstraction libraries like LangChain.  
Just Hugging Face transformers and models from Hugging Face.  
Runs in 12GB VRAM.  

## Files:

- **tools.py** : Python file with helper functions.
- **customEmbedFunction.py** : Python file with ChromaDB custom embed function class. Default model: Hugging Face `w601sxs/b1ade-embed`.
- **pdf2md_llama_parse.ipynb** : Jupyter notebook for parsing complex PDFs into markdown files (uses LLama Parse).
- **md2chromadb.ipynb** : Jupyter notebook for inserting and updating embeddings from markdown files into local ChromaDB vector store (with custom embedding function).
- **simple-rag.ipynb** : Jupyter notebook for the RAG system (same embedder as retriever and `microsoft/Phi-3-mini-128k-instruct` as LLM).

Example used:  
Ask clarification of board game rules based on the official rulebook PDF.  

## Future development:  

- Streamlit UI
- FastAPI

