from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# connect to your document database
persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# setup AI model
model = ChatOpenAI(model="gpt-5-nano")

# store our conversation as message
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        message = [
            SystemMessage(content="given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.")
        ] + chat_history + [
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = model.invoke(message)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # find relevant documents
    retriever = db.as_retriever(search_kwargs={'k':3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # show first two lines of each documents
        lines = doc.page_content.split('\n')[:2]
        preview = "\n".join(lines)
        print(f"Document {i}: {preview}...")

    # step 3: create a final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    
    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}
    
    please provide a clear, helpful answers using only the information from these documents.
    If you can find the answer from the provided documents, say "i don't have enough information to answer your question based on the provided documents."
"""

    # step 4: get the answer
    message = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(message)
    answer = result.content

    # Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# simple chat loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour Question: ")

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        print(chat_history)

        ask_question(user_question=question)

if __name__ == "__main__":
    start_chat()


# "C:\Users\Admin\Data Science\rag\.venv\Scripts\python.exe" "C:\Users\Admin\Data Science\rag\04_history_aware_generation.py"
# Ask me questions! Type 'quit' to exit.
#
# Your Question: what is the latest projects in google
# []
#
# --- You asked: what is the latest projects in google ---
# Found 3 relevant documents:
# Document 1: Google for Jobs is an enhanced search feature that aggregates listings from job boards and career
# sites.[183] Google Earth, launched in 2005, allows users to see high-definition satellite pictures from all...
# Document 2: In 2008, Google announced its "project 10100", which accepted ideas for how to help the community and
# then allowed Google users to vote on their favorites.[350] After two years of no update, during which...
# Document 3: and Gemini), machine learning APIs (TensorFlow), AI Ruth Porat (President and
# chips (TPU), and more. Many of these products and CIO)...
# Answer: The documents do not specify a single, clearly defined “latest” Google project. They mention several ongoing or recently noted initiatives, including:
#
# - Gemini – mentioned as a recent AI-related project/product.
# - Google DeepMind transformer models – AI research and models.
# - TensorFlow – machine learning APIs, with TPU chips referenced.
# - Waymo – self-driving cars.
# - Sidewalk Labs – smart cities initiative.
# - Sycamore – quantum computing.
# - Nest – smart home devices and related ventures.
#
# Other items in the documents include broad products and services (e.g., Android, Google Earth, Google for Jobs, Chrome, ChromeOS) and a reference to Project 10100 from 2008. Some previously launched products (Stadia) and others (Google+, Reader, Hangouts, Inbox by Gmail) are noted as discontinued.
#
# In short, the documents list several notable projects and initiatives, with Gemini being one of the most recent AI-related items mentioned, but they do not designate a single, official “latest” Google project. If you need the current, up-to-date list, more recent sources would be required.
#
# Your Question: what is the most advanced ai project they are doing now
# [HumanMessage(content='what is the latest projects in google', additional_kwargs={}, response_metadata={}), AIMessage(content='The documents do not specify a single, clearly defined “latest” Google project. They mention several ongoing or recently noted initiatives, including:\n\n- Gemini – mentioned as a recent AI-related project/product.\n- Google DeepMind transformer models – AI research and models.\n- TensorFlow – machine learning APIs, with TPU chips referenced.\n- Waymo – self-driving cars.\n- Sidewalk Labs – smart cities initiative.\n- Sycamore – quantum computing.\n- Nest – smart home devices and related ventures.\n\nOther items in the documents include broad products and services (e.g., Android, Google Earth, Google for Jobs, Chrome, ChromeOS) and a reference to Project 10100 from 2008. Some previously launched products (Stadia) and others (Google+, Reader, Hangouts, Inbox by Gmail) are noted as discontinued.\n\nIn short, the documents list several notable projects and initiatives, with Gemini being one of the most recent AI-related items mentioned, but they do not designate a single, official “latest” Google project. If you need the current, up-to-date list, more recent sources would be required.', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[])]
#
# --- You asked: what is the most advanced ai project they are doing now ---
# Searching for: What is Google's most advanced AI project currently underway?
# Found 3 relevant documents:
# Document 1: Following the success of ChatGPT and concerns that Google was falling behind in the AI race, Google's
# senior management issued a "code red"[132] and a "directive that all of its most important products—those...
# Document 2: 20  percent of clicks were fraudulent or invalid.[160] Google Search Console (rebranded from Google
# Webmaster Tools in May 2015) allows webmasters to check the sitemap, crawl rate, and for security...
# Document 3: In July 2025, the United States Department of Defense announced that Google had received a $200
# million contract for AI in the military, along with Anthropic, OpenAI, and xAI.[143]...
# Answer: Gemini (formerly Bard) — Google's generative AI chatbot, described in the documents as the project seen as a legitimate competitor to ChatGPT and the most advanced AI effort mentioned. It started as Bard in 2023 and was rebranded Gemini in 2024.
#
# Your Question: quit
# Goodbye!