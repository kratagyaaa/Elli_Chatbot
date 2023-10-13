from PyPDF2 import PdfReader  # Import PdfReader instead of PdfFileReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from transformers import GPT2TokenizerFast
from pathlib import Path
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

import json
from flask import Flask, jsonify, request

import re
from datetime import datetime

_ = load_dotenv(find_dotenv())

file_path = "./AHC_Web_Scraping.pdf"

# Step 1: Extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file_path):
    pdf_reader = PdfReader(pdf_file_path)  # Use PdfReader instead of PdfFileReader
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from the PDF file
doc = extract_text_from_pdf(file_path)

# Save the extracted text to a text file with UTF-8 encoding
with open("extracted_text.txt", "w", encoding="utf-8") as text_file:
    text_file.write(doc)

with open("extracted_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less
type(chunks[0])

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Check similarity search is working
query = "Who created transformers?"
docs = db.similarity_search(query)

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

query = "tell me about the masterclass in detail?"
docs = db.similarity_search(query)

# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())
chat_history = []

# Setup app
app = Flask(__name__)

@app.route("/test")
def home():
    return "Hello, :)"

@app.route("/setup", methods=["GET"])
def setup_server():
    if request.method == "GET":
        try:
            chain.run(input_documents=docs, question=query)
            return (
                jsonify(
                    {"message": "Server Setup Successfully completed.", "status": 200}
                ),
                200,
            )
        except Exception as e:
            return jsonify({"message": "Internal server error.", "status": 500}), 500
    else:
        return jsonify({"message": "Bad request.", "status": 400}), 400

@app.route("/chatbot", methods=["POST"])
def run_chatbot():
    if request.method == "POST":
        data = json.loads(request.data)
        queryValue = data["query"]
        try:
            query = queryValue
            print("query: ", query)

            if query.lower() == "exit":
                return (
                    jsonify(
                        {
                            "message": "Thank you for using the State of the Union chatbot!",
                            "status": 200,
                        }
                    ),
                    200,
                )

            result = qa({"question": query, "chat_history": chat_history})

            return jsonify({"message": result["answer"], "status": 200}), 200
        except Exception as e:
            return jsonify({"message": "Bad request.", "status": 400}), 404

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
