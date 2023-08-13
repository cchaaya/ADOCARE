"""## Libraries"""

# Set the path to the ffmpeg executable
# os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"

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

# Get the total number of loaded PDFs in the 'pdfs' list
# len(pdfs)

# Retrieve the content of the PDF at index 40 from the 'pdfs' list
# page = pdfs[40]

# Access the metadata of the loaded PDF page
# page.metadata

# Print the content of the loaded PDF page
# print(page.page_content)

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
# print (len(docs))

# print the loaded audio content
# docs[0].page_content[0:500]

# """## URLs"""

# Create a web loader instance to load content from the specified URL
web_loader = WebBaseLoader("https://www.webmd.com/teens/how-to-tell-parents-pregnant")

# Load the content of the webpage using the web loader
webpages = web_loader.load()

# check the number of loaded documents
# len(webpages)

# Print the first loaded document
# webpages[0].page_content

# """## Final Corpus"""

# Create an empty list to store the combined corpus
corpus = []

# Add documents from each source to the corpus list
corpus.extend(pdfs)
# corpus.extend(docs)
corpus.extend(webpages)

# print(len(corpus))

# len(pdfs)+len(docs)+len(webpages)

# """# Document Splitting

# ## Recursive Character Text Splitter
# """

# Create a RecursiveCharacterTextSplitter instance
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Maximum size of each text chunk in characters
    chunk_overlap=150,      # Number of characters overlapping between adjacent chunks
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]  # List of separators used to split the text
)

# Use the RecursiveCharacterTextSplitter to split the 'corpus' into smaller pages
rec_splitted_pages = r_splitter.split_documents(corpus)

# print the number of documents
# print(len(rec_splitted_pages))

# print the content of the selected page
# rec_splitted_pages[51]

# """# Vectorstores and Embeddings

# """

# Create an instance of the OpenAIEmbeddings class
embedding = OpenAIEmbeddings()

# Define the directory path for data persistence related to chroma
persist_directory = 'docs/chroma/'

# remove old database files if any
# !rm -rf ./docs/chroma

# Create a Chroma instance named 'vectordb' using embeddings and documents
vectordb = Chroma.from_documents(
    documents=rec_splitted_pages,        # Collection of segmented documents
    embedding=embedding,                 # The embedding model to use
    persist_directory=persist_directory  # Directory for storing the vector database
)

# Print the count of documents in the Chroma vector database
# print(vectordb._collection.count())

# """# Retrieval Methods

# ## Query
# """

# # Define the question
# question = "how can I prevent pregnancy?"

# """## Similarity Search

# *   focuses solely on semantic similarity between documents and the query, aiming to provide the most relevant matches.
# # """

# # Perform a similarity search using the Chroma vector database
# ss_docs = vectordb.similarity_search(question,k=3) # k is number of documents

# # calculates the number of retrieved documents
# # len(ss_docs)

# # Print the content of the n retrieved document
# # ss_docs[0].page_content
# # ss_docs[1].page_content
# # ss_docs[2].page_content

# ss_docs[1].page_content

# """## Addressing Diversity: Maximum marginal relevance

# *   emphasizes diversity by selecting documents that maintain a balance between relevance and dissimilarity, making it suitable when a varied set of informative documents is desired.
# """

# Perform a Maximal Marginal Relevance (MMR) search using the Chroma vector database
# mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)

# Print the first 500 characters of the content of the n MMR-retrieved document
# mmr_docs[0].page_content[:500]
# mmr_docs[1].page_content[:500]
# mmr_docs[2].page_content[:500]

# mmr_docs[1].page_content[:500]

# mmr_docs[2].page_content[:500]

# """# Retrieval Chain

## Specify the LLM name and version
# """

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
# """

# Create a RetrievalQA instance named 'qa_chain' with an LLM and MMR-based retriever
qa_chain = RetrievalQA.from_chain_type(
    llm,  # OpenAI Language Model instance
    retriever=vectordb.as_retriever(search_type="mmr")  # MMR-based retrieval component
)

# Perform question-answering using the 'qa_chain' instance with the given query
# result = qa_chain({"query": question})

# # print the result
# result["result"]

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

# # Define the question
# question = "When did WWII happen?"

# # Use the configured CRC instance to perform question-answering with the given query
# result = p_qa_chain({"query": question})

# result["result"]

# """## Conversational Retrieval Chain

# *   Create a Conversational Retrieval Chain (CRC) by combining the power of an OpenAI Language Model (LLM) with a a selected retrieval component (SS or MMR)
# * As a main components of the CRC, use the prompt component to guide the
# conversation to extract data from our corpus only and configure a memory mechanism to store and manage the contextual information throughout the interaction.

### Chat (with memory)
# """

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

# # Define the question
# # question = "Name 3 ways to prevent pregnancy?"
# question = "how do i tell my parents that i am pregnant?"

# result = conv_qa({"question": question})

# # Print the results
# result['answer']

# # Define the question to test the memory
# # question = "can you tell me again what is the third one?"
# question = "and will they understand?"
# result = conv_qa({"question": question})
# result['answer']

# """# Chatbot application
"""Now, it's your trun to try with any questions you have!
# """

# Conversational Retrieval Chain implementation
# if __name__ == "__main__":
#     while True:
#         # question = input("Please enter your question:")
#         user_question = input("\033[1mUser question:\033[0m ")
#         if user_question.lower() == "exit":
#             print("Goodbye!")
#             break

#         # Perform conversational question-answering using the 'conv_qa' instance
#         result = conv_qa({"question": user_question})

#         # Print the bold label and the generated answer
#         print("\033[1mAdocare reply:\033[0m", result['answer'])


# def main():
#     st.title("Adocare Chatbot")

#     user_question = st.text_input("User question:")

#     if user_question:
#         # Perform conversational question-answering using the model
#         result = conv_qa({"question": user_question})

#         # Display the question and answer in a fixed-size text area
#         st.text("User question: " + user_question)
#         st.text_area("Adocare reply:", result['answer'], height=200)

# if __name__ == "__main__":
#     main()

# Clear Streamlit cache
st.set_option('deprecation.showfileUploaderEncoding', False)

import streamlit as st

def main():
    st.title("Adocare Chatbot")

    conversation = []

    with st.form("user_input_form"):
        user_question = st.text_input("User question:")

        if st.form_submit_button(label="Submit") and user_question:
            # Perform conversational question-answering using the model
            result = conv_qa({"question": user_question})

            # Append the user's prompt and the bot's reply to the conversation list
            conversation.append(("User:", user_question))
            conversation.append(("Adocare:", result['answer']))

    # Display the conversation history in reverse order
    conversation_display = "\n".join([f"{sender} {message}" for sender, message in reversed(conversation)])
    st.text(conversation_display)

if __name__ == "__main__":
    main()










