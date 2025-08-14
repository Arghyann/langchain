from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os

# Load the restaurant reviews
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"

# Check if we need to create a new vector store
add_documents = not os.path.exists(db_location)

if add_documents:
    # Create document objects from reviews
    documents = []
    ids = []
    for index, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(index)
        )
        ids.append(str(index))
        documents.append(doc)

# Initialize the vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddings,
    persist_directory=db_location
)

# Add documents to the vector store if needed
if add_documents:
    vector_store.add_documents(documents, ids=ids)

# Create a retriever that returns the top 5 most relevant reviews
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)