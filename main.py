import os
import requests
import tempfile
import uvicorn
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

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

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[AnswerResponse] = None

class JobCreationResponse(BaseModel):
    job_id: str

jobs: Dict[str, Dict] = {}

app = FastAPI(
    title="AI Document Q&A Processor API",
    description="An API that processes a document to answer questions in a single batch asynchronously.",
    version="2.0.0"
)

def process_document_logic(document_url: str, questions: List[str]) -> AnswerResponse:
    tmp_file_path = None
    try:
        print(f"Downloading document from: {document_url}")
        response = requests.get(str(document_url))
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})

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
        Return your answers in a single JSON object that strictly follows this format: {json_format_instructions}
        CONTEXT:\n---\n{context}\n---\nQUESTIONS_LIST:\n{questions}
        """
        prompt = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "questions"],
            partial_variables={"json_format_instructions": parser.get_format_instructions()}
        )
        chain = (
            {"context": (lambda x: " ".join(x["questions"])) | retriever,
             "questions": (lambda x: x["questions"])}
            | prompt | llm | parser
        )
        print(f"Processing all {len(questions)} questions in a single batch...")
        llm_response = chain.invoke({"questions": questions})
        
        return AnswerResponse(
            source_document=str(document_url),
            answers=llm_response.get('answers', [])
        )
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

async def run_background_task(job_id: str, request: QuestionRequest):
    try:
        result = await asyncio.to_thread(process_document_logic, request.documents, request.questions)
        jobs[job_id]['status'] = 'complete'
        jobs[job_id]['result'] = result
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['result'] = {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=JobCreationResponse, status_code=202)
async def run_adjudication(
    request: QuestionRequest,
    authorization: Optional[str] = Header(None)
):
    api_key = os.getenv("API_KEY")
    if not api_key or authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key.")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "result": None}
    asyncio.create_task(run_background_task(job_id, request))
    
    return {"job_id": job_id}

@app.post("/hackrx/run_sync", response_model=AnswerResponse)
async def run_sync(
    request: QuestionRequest,
    authorization: Optional[str] = Header(None)
):
    api_key = os.getenv("API_KEY")
    if not api_key or authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key.")
    
    try:
        result = await asyncio.to_thread(process_document_logic, request.documents, request.questions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.get("/status/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, status=job['status'], result=job.get('result'))

@app.get("/")
def read_root():
    return {"message": "AI Document Q&A Processor API is running."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
