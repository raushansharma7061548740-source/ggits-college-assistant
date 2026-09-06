from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

# Import your compiled LangGraph
from main import app as langgraph_app


# -----------------------------------
# Create FastAPI application
# -----------------------------------

api = FastAPI(
    title="GGITS Assistant API",
    description="College Assistant powered by LangGraph and RAG",
    version="1.0.0"
)


# -----------------------------------
# Enable CORS
# -----------------------------------

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# Request Model
# -----------------------------------

class ChatRequest(BaseModel):
    message: str
    programme: str
    session_id: Optional[str] = None


# -----------------------------------
# Response Model
# -----------------------------------

class ChatResponse(BaseModel):
    reply: str
    category: str
    session_id: str


# -----------------------------------
# Chat API
# -----------------------------------

@api.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        # Create session ID if frontend doesn't have one
        session_id = request.session_id

        if not session_id:
            session_id = str(uuid4())


        # Invoke your LangGraph
        result = langgraph_app.invoke({

            "programme": request.programme,

            "message": [
                ("human", request.message)
            ]

        })


        # Get final AI response
        reply = result["message"][-1].content


        # Get classifier category
        category = result.get("query_type", "general")


        # Return data to frontend
        return ChatResponse(
            reply=reply,
            category=category,
            session_id=session_id
        )


    except Exception as e:

        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -----------------------------------
# Health Check API
# -----------------------------------

@api.get("/")
def home():

    return {
        "message": "GGITS Assistant API is running successfully"
    }


@api.get("/health")
def health():

    return {
        "status": "healthy"
    }