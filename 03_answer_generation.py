from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

# load embedding and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# search for relevant document
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs = {
#         "k":3,
#         "score_threshold":0.3   # Only return chunks with cosine similarity ≥ 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

# print(f"User Query: {query}")
# # Display results
# print("--- Context ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")

# combine the query and relevant documents

combined_input = f"""
Based on the following documents please answer this question: {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. if you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents." 
"""

# create a ChatOpenAI model
model = ChatOpenAI(model="gpt-5-nano")

# Define the message for the model
message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# invoke the model with combined input
result = model.invoke(message)

# display the full result and content only
print("---  Generated Response  ---")
# print("FUll Result: ")
# print(result)
print("Content Only: ")
print(result.content)