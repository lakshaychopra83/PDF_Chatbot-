from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat PDF 💬")

    # Access OpenAI API Key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key is None:
        raise ValueError("OpenAI API key is missing. Set the OPENAI_API_KEY environment variable.")

    # Upload file
    pdf = st.file_uploader("Upload your PDF file", type="pdf")

    # Extract text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = char_text_splitter.split_text(text)

        # Create embeddings
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAIEmbeddings: {str(e)}")
            return

        # Vectorize the text using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(text_chunks)

        # Show user input
        query = st.text_input("Type your question:")
        if query:
            # Find the most similar document
            query_vector = vectorizer.transform([query])
            similarity_scores = cosine_similarity(tfidf_matrix, query_vector)
            most_similar_index = similarity_scores.argmax()
            most_similar_doc = text_chunks[most_similar_index]

            # Create a Document instance
            most_similar_document = Document(page_content=most_similar_doc, metadata={})

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            # Pass the Document instance to the question-answering chain
            response = chain.run(input_documents=[most_similar_document], question=query)
            st.write(response)

if __name__ == '__main__':
    main()
