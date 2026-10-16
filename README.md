# PDF RAG Demo

A teaching project with Streamlit, local sentence-transformer embeddings, persistent Chroma vector storage, and Groq for answers.

## Setup (Windows PowerShell, Python 3.10+)

Open a terminal in this folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Edit `.env` and replace `your_groq_api_key_here` with your key from https://console.groq.com/keys. The key is loaded from this folder regardless of the terminal's current directory. Never share `.env`. You can change `GROQ_MODEL` to another model available in your Groq account.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal. Upload a text-based PDF, click **Build PDF index**, then ask a question. Answers include page citations; expand the evidence panel to inspect retrieved excerpts. Questions are answered independently; displayed chat history is not passed to the model.

## Notebook walkthrough

Open `RAG_Demo.ipynb` in Jupyter or VS Code and select the project environment. Run cells in order. Set `PDF_PATH` to your PDF. The notebook demonstrates extraction, chunks, embeddings, Chroma retrieval, and the Groq response. The Python files also use `# %%` notebook-style cells.

```powershell
.\.venv\Scripts\python.exe -m notebook RAG_Demo.ipynb
```

## How it works

1. pypdf extracts text with original PDF page numbers.
2. Text is split into 1,000-character chunks with 150-character overlap.
3. `all-MiniLM-L6-v2` embeds chunks locally.
4. Chroma saves vectors, text, and page metadata in `chroma_db/`.
5. A question is embedded and its nearest chunks are retrieved.
6. Groq receives the question and excerpts and is instructed to cite pages or admit missing evidence.

First use needs internet to download the embedding model; Groq calls also require internet. Uploaded text stays in the local database, but retrieved excerpts are sent to Groq. Clearing chat or changing the upload does not erase saved indexes. Stop the app and remove `chroma_db/` if you want to delete all saved indexes.

This is a local classroom demo for one PDF at a time. Scanned/image PDFs need OCR beforehand. Retrieval can miss relevant information, and model citations should be checked against the excerpts. It is not intended as a multi-user production service.

API references: [Groq SDK](https://github.com/groq/groq-python), [Chroma](https://docs.trychroma.com/reference/python), [Streamlit](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader).
