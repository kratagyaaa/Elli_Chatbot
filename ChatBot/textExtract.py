# Advanced method - Split by chunk
# textExtract.py
# Step 1: Convert PDF to text
from matplotlib import pyplot as plt
import textract
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

_ = load_dotenv(find_dotenv())

# base_path = Path(__file__).parent
# file_path = (base_path / "data/Ortal.pdf").resolve()
# print(file_path)
# file_path = "D:/workspace/Clients/Sameeskha-Manish/python/APIs/data/Ortal.pdf"
file_path = "./Ortal.pdf"


# myFilePath = os.environ["FILE_PATH"]
# print("myFilePath:", myFilePath)

# path = Path(file_path)
# file_path = "data\Ortal.pdf"
# doc = textract.process(myFilePath)
doc = textract.process(file_path)


# print("Doc: ", doc)

# Step 2: Save to .txt and reopen (helps prevent issues)
with open("attention_is_all_you_need.txt", "w") as f:
    f.write(doc.decode("utf-8"))

with open("attention_is_all_you_need.txt", "r") as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0])

# # Quick data visualization to ensure chunking was successful

# # Create a list of token counts
# token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

# # Create a DataFrame from the token counts
# df = pd.DataFrame({'Token Count': token_counts})

# # Create a histogram of the token count distribution
# df.hist(bins=40, )

# # Show the plot
# plt.show()

# 2. Embed text and store embeddings

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)


# 3. Setup retrieval function
# Check similarity search is working
query = "Who created transformers?"
docs = db.similarity_search(query)
# print(docs[0])


# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

query = "tell me about the masterclass in detail?"
docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)

# print(chain)


# 5. Create chatbot with chat memory (OPTIONAL)
from IPython.display import display
import ipywidgets as widgets

# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())
chat_history = []


def on_submit(_):
    query = input_box.value
    input_box.value = ""

    if query.lower() == "exit":
        print("Thank you for using the State of the Union chatbot!")
        return

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))

    display(widgets.HTML(f"<b>User:</b> {query}"))
    display(
        widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}')
    )


print("Welcome to the Transformers chatbot! Type 'exit' to stop.")

input_box = widgets.Text(placeholder="Please enter your question:")
input_box.on_submit(on_submit)

display(input_box)
