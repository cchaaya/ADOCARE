"""## Libraries"""
import os
import openai
import sys
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import datetime
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import yt_dlp
import streamlit as st


# This will read and set environment variables from the .env file in the Colab environment
load_dotenv()

# Set the OpenAI API key for authentication
# os.environ['OPENAI_API_KEY'] = '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'
# os.environ['OPENAI_API_KEY'] = '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'
# os.environ['OPENAI_API_KEY'] = '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'
# os.environ['OPENAI_API_KEY'] = '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'
os.environ['OPENAI_API_KEY'] = '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'

## PDFs
# The code initializes PDF loaders and loads multiple PDF documents into a list
pdf_loaders = [
    PyPDFLoader("12QAsSRH.pdf"),
    #PyPDFLoader("//content/SexualHarassment.pdf"),
    #PyPDFLoader("/content/SexuallyTransmittedDiseases.pdf")
]
pdfs = []
for pdf_loader in pdf_loaders:
    pdfs.extend(pdf_loader.load())


# """## YouTube"""

# # Set the OpenAI API key for authentication
# openai.api_key = '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'

# # Define the URL of the YouTube video and the directory to save the content
# url="https://www.youtube.com/watch?v=umpBnIxOqy8"
# save_dir="docs/youtube/"

# # Create a loader using a combination of YouTube audio loader and OpenAI Whisper parser
# loader = GenericLoader(
#     YoutubeAudioLoader([url],save_dir),
#     OpenAIWhisperParser()
# )

# Load the audio content and parse it
# docs = loader.load()

# """## URLs"""

# Create a web loader instance to load content from the specified URL
web_loader = WebBaseLoader("https://www.webmd.com/teens/how-to-tell-parents-pregnant")

# Load the content of the webpage using the web loader
webpages = web_loader.load()

# """## Final Corpus"""

# Create an empty list to store the combined corpus
corpus = []

# Add documents from each source to the corpus list
corpus.extend(pdfs)
# corpus.extend(docs)
corpus.extend(webpages)


# """# Document Splitting

### Recursive Character Text Splitter

# Create a RecursiveCharacterTextSplitter instance
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum size of each text chunk in characters
    chunk_overlap=150,      # Number of characters overlapping between adjacent chunks
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]  # List of separators used to split the text
)

# Use the RecursiveCharacterTextSplitter to split the 'corpus' into smaller pages
rec_splitted_pages = r_splitter.split_documents(corpus)

# """# Vectorstores and Embeddings

# Create an instance of the OpenAIEmbeddings class
embedding = OpenAIEmbeddings()

# Define the directory path for data persistence related to chroma
persist_directory = 'docs/chroma/'

# Create a Chroma instance named 'vectordb' using embeddings and documents
vectordb = Chroma.from_documents(
    documents=rec_splitted_pages,        # Collection of segmented documents
    embedding=embedding,                 # The embedding model to use
    persist_directory=persist_directory  # Directory for storing the vector database
)

# Print the count of documents in the Chroma vector database
# print(vectordb._collection.count())

# """# Retrieval Chain

## Specify the LLM name and version

# Get the current date
current_date = datetime.datetime.now().date()

# Check if the current date is before September 2, 2023
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
# Print the current date
# print(current_date)
# Print the selected LLM name
# print(llm_name)

# Create an instance of the ChatOpenAI class with the specified LLM name and temperature
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# """## RetrievalQA chain

# *   Create a simple QA retreival chain with LLM and selected retreival component


# Create a RetrievalQA instance named 'qa_chain' with an LLM and MMR-based retriever
qa_chain = RetrievalQA.from_chain_type(
    llm,  # OpenAI Language Model instance
    retriever=vectordb.as_retriever(search_type="mmr")  # MMR-based retrieval component
)


# """### QA Retrieval chain with Prompt"""

# Build a prompt template for question-answering
template = """Answer the question using information from our corpus only. If you don't know the answer, just say that "I don't know the answer to this question." Don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
# Create a PromptTemplate instance from the template
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run the QA retrieval chain with the defined prompt
p_qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_type="mmr"),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


### Chat (with memory)

# Create a ConversationBufferMemory instance for managing conversational context
memory = ConversationBufferMemory(
    memory_key="chat_history",    # Name/key for the memory component
    return_messages=True          # Setting to return messages along with other data
)

# """### Run the Conversational Retrieval Chain with the Prompt & Memory components"""

# Create a ConversationalRetrievalChain instance for conversational question-answering
conv_qa = ConversationalRetrievalChain.from_llm(
    llm,                                           # OpenAI Language Model instance
    retriever=vectordb.as_retriever(search_type="mmr"),  # MMR-based retriever component
    memory=memory,                                 # ConversationBufferMemory for context
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}  # Prompt for document combining
)


# """# Chatbot application

# st.title("Adocare Chatbot")
# st.subheader("Feel free to ask any questions you have!")
# st.markdown("This is a demo Chatbot designed to respond to adolescents' inquiries related to sexual and reproductive health."
#             " Ongoing development is being made for further enhancement")

def main():
    conversation = []  # Initialize an empty list to store the conversation history

    # Add the title and image next to each other
    col1, col2, col3 = st.columns([1, 1, 1])  # Divide the screen into three equal-width columns
    with col1:
        st.title("Adocare Chatbot")  # Display the title
    with col2:
        st.write("")  # Empty column for spacing
    with col3:
        st.image("Group_pic.JPG", use_column_width=True, width=300)  # Display the image with increased width

    st.subheader("Feel free to ask any questions you have!")
    st.markdown("This is a demo Chatbot designed to respond to adolescents' inquiries related to sexual and reproductive health."
                " Ongoing development is being made for further enhancement")

    with st.form("user_input_form"):
        user_question = st.text_input("User question:")

        if st.form_submit_button(label="Submit") and user_question:
            # Perform conversational question-answering using the model
            result = conv_qa({"question": user_question})

            # Append the user's prompt and the bot's reply to the conversation list
            conversation.append(("User:", user_question))
            conversation.append(("Adocare:", result['answer']))

            # Clear the user's input after submitting
            user_question = ""  # Set the user_question variable to an empty string

    # Display the conversation history in reverse order
    conversation_display = "\n".join([f"{sender} {message}" for sender, message in conversation])
    st.text_area("Conversation History:", conversation_display, height=200)  # Display the conversation history

if __name__ == "__main__":
    main()



