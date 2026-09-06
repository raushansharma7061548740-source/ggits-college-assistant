import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


ACADEMIC_PDF_URL = (
    "https://raw.githubusercontent.com/"
    "raushansharma7061548740-source/Academic.pdf/"
    "main/academics_handbook.pdf"
)

FEE_PDF_URL = (
    "https://raw.githubusercontent.com/"
    "raushansharma7061548740-source/fee_structure.pdf/"
    "main/fee_structure.pdf"
)


# step-1 building the rag retrievers

import os
import tempfile
import requests

from langchain_community.document_loaders import PyPDFLoader


def build_retriever_from_url(pdf_url: str):

    print(f"Downloading PDF from GitHub...")

    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as temp_file:

        temp_file.write(response.content)

        temp_pdf_path = temp_file.name

    try:

        loader = PyPDFLoader(temp_pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

    finally:

        # Delete temporary PDF after FAISS is created
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


academic_retriever = build_retriever_from_url(ACADEMIC_PDF_URL)
fee_retriever = build_retriever_from_url(FEE_PDF_URL)

print("RAG system ready!")


llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0.4
)

# step2 - state
class State(TypedDict):
    programme : str
    message : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str

# step 3 - Nodes Generation

def classifier_node(state : State)-> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['message'][-1].content
    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"

        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"

        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"

        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    print(f"[DEBUG] raw classifier output: {response.content!r}")
    print(f"[DEBUG] final category: {category}")

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state : State) -> dict:
    """Retrievers relevent chunks from the academic handbook."""
    query = state['message'][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return{"retrieved_context": context}



def fee_rag_node(state : State) -> dict:
    """Retrievers relevent chunks from the fee structure PDF."""
    query = state['message'][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return{"retrieved_context": context}



def general_node(state : State) -> dict:
    """Answers directly using the LLm's own knowledge, no retrieval needed."""
    return {"retrieved_context" : "NO_RETRIEVAL_NEEDED"}



def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["message"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"message": [("ai", response.content.strip())]}


# step 4 - router function

def route_query(state : State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == 'fee':
        return "fee_rag"
    else:
        return "general"

# step 5 - Building the graph

graph = StateGraph(State)

graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)


# edges

graph.add_edge(START,"classifier")
graph.add_conditional_edges(
    "classifier",route_query
)

graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")


graph.add_edge("response",END)

from langgraph.checkpoint.memory import MemorySaver



memory = MemorySaver()

app = graph.compile(
    checkpointer=memory
)
