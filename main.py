import os
import requests
import tempfile
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# --- LangChain and AI Components ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Pydantic Models for LLM and API Response ---
class QAPair(BaseModel):
    question: str = Field(description="The original question that was asked.")
    answer: str = Field(description="The answer to the question based on the provided context.")

class AnswerList(BaseModel):
    answers: List[QAPair]

class QuestionRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    source_document: str
    answers: List[QAPair]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Document Q&A Processor API",
    description="An API that processes a document to answer questions in a single batch.",
    version="1.2.0" # Version bump
)

# --- Core RAG Logic ---
def get_qa_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=os.getenv("GOOGLE_API_KEY"), 
        temperature=0
    )
    parser = JsonOutputParser(pydantic_object=AnswerList)
    rag_prompt_template = """
    You are an expert assistant. Your task is to answer all the user's questions based on the single context provided below.
    Analyze the context and provide a clear and concise answer for each question in the list.
    If the context does not contain the answer for a specific question, state 'The provided document does not contain information on this topic.' for that question's answer.

    Return your answers in a single JSON object that strictly follows this format:
    {json_format_instructions}

    CONTEXT:
    ---
    {context}
    ---

    QUESTIONS_LIST:
    {questions}
    """
    prompt = PromptTemplate(
        template=rag_prompt_template,
        input_variables=["context", "questions"],
        partial_variables={"json_format_instructions": parser.get_format_instructions()}
    )
    chain = (
        {
            "context": (lambda x: " ".join(x["questions"])) | retriever,
            "questions": (lambda x: x["questions"]),
        }
        | prompt
        | llm
        | parser
    )
    return chain

# --- API Endpoints ---

# NEW: Health Check Endpoint for Debugging
@app.get("/health")
def health_check():
    """
    A simple endpoint to verify that the server is running.
    If you can reach this, the server is up. If not, it's likely crashing.
    """
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=AnswerResponse)
def run_adjudication(
    request: QuestionRequest,
    authorization: Optional[str] = Header(None)
):
    api_key = os.getenv("API_KEY")
    if not api_key or authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key.")

    tmp_file_path = None
    try:
        print(f"Downloading document from: {request.documents}")
        response = requests.get(str(request.documents))
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        print(f"Document saved to temporary file: {tmp_file_path}")

        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        print("Splitting documents and creating vector store...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})

        qa_chain = get_qa_rag_chain(retriever)
        
        print(f"\nProcessing all {len(request.questions)} questions in a single batch...")
        llm_response = qa_chain.invoke({"questions": request.questions})
        print(f"Generated Answers (batched): {llm_response}")
        
        return AnswerResponse(
            source_document=str(request.documents),
            answers=llm_response.get('answers', [])
        )

    except requests.exceptions.RequestException as e:
        print(f"Error downloading document: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download or access the document URL: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")

@app.get("/")
def read_root():
    return {"message": "AI Document Q&A Processor API is running."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
