# %% [markdown]
# # Streamlit PDF question-answering demo
# Run with: python -m streamlit run app.py

# %% Imports
import hashlib
import streamlit as st
from rag_pipeline import load_embedder, index_pdf, retrieve, answer_question

# %% Page and cached model
st.set_page_config(page_title='PDF RAG Demo', page_icon='📚', layout='wide')
st.title('📚 Ask your PDF')
st.caption('Upload a PDF, build its vector index, and ask questions grounded in its text.')

@st.cache_resource
def get_embedder():
    return load_embedder()

with st.sidebar:
    st.header('1. Upload a PDF')
    uploaded = st.file_uploader('Choose a PDF', type=['pdf'])
    top_k = st.slider('Excerpts to retrieve', 1, 8, 4)
    st.caption('Embeddings and Chroma storage run locally. Retrieved excerpts and your question are sent to Groq.')
    st.caption('First indexing downloads an embedding model. Scanned PDFs need OCR.')

# %% Reset chat whenever the selected PDF changes
pdf_bytes = uploaded.getvalue() if uploaded else None
selected_id = hashlib.sha256(pdf_bytes).hexdigest() if pdf_bytes else None
if st.session_state.get('selected_id') != selected_id:
    st.session_state.selected_id = selected_id
    st.session_state.messages = []
    st.session_state.pop('collection', None)
st.session_state.setdefault('messages', [])

if uploaded:
    if st.sidebar.button('Build PDF index', type='primary'):
        try:
            with st.spinner('Extracting text and creating the vector index…'):
                collection, pages, chunks = index_pdf(pdf_bytes, get_embedder())
            st.session_state.collection = collection
            st.session_state.stats = (pages, chunks)
            st.session_state.messages = []
        except Exception as exc:
            st.error(f'Could not index the PDF: {exc}')
else:
    st.info('Upload a PDF in the sidebar to begin.')

# %% Chat and source evidence
if 'collection' in st.session_state:
    pages, chunks = st.session_state.stats
    st.success(f'Ready: {uploaded.name} · {pages} pages · {chunks} chunks')

def show_sources(sources):
    with st.expander('View retrieved PDF excerpts'):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Excerpt {i} — Page {source['page']}**")
            st.write(source['text'])

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            show_sources(message['sources'])

question = st.chat_input('Ask a question about this PDF',
                         disabled='collection' not in st.session_state)
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Finding relevant excerpts and generating an answer…'):
                sources = retrieve(question, st.session_state.collection, get_embedder(), top_k)
                answer = answer_question(question, sources)
            st.markdown(answer)
            show_sources(sources)
            st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': sources})
        except Exception as exc:
            st.error(f'Could not answer: {exc}')
