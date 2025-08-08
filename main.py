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
# It's good practice to group related imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Pydantic Models for LLM and API Response ---
# These define the data structures for your API and the LLM's expected output.

class QAPair(BaseModel):
    """A model for a single question and its corresponding answer."""
    question: str = Field(description="The original question that was asked.")
    answer: str = Field(description="The answer to the question based on the provided context.")

class AnswerList(BaseModel):
    """A model for a list of QAPair objects, used for the LLM's structured output."""
    answers: List[QAPair]

class QuestionRequest(BaseModel):
    """Defines the structure of the incoming API request body."""
    documents: HttpUrl
    questions: List[str]

class AnswerResponse(BaseModel):
    """Defines the structure of the successful API response."""
    source_document: str
    answers: List[QAPair]

# --- FastAPI App Initialization ---

app = FastAPI(
    title="AI Document Q&A Processor API",
    description="An API that processes a document to answer questions in a single batch.",
    version="1.1.0"
)

# --- Core RAG Logic ---

def get_qa_rag_chain(retriever):
    """Initializes and returns a RAG chain optimized for batch question answering."""
    # Initialize the Google Generative AI model
    # Ensure GOOGLE_API_KEY is set in your .env file
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        google_api_key=os.getenv("GOOGLE_API_KEY"), 
        temperature=0
    )
    
    # The parser expects the LLM to return a JSON object matching the AnswerList model.
    parser = JsonOutputParser(pydantic_object=AnswerList)

    # This prompt template is structured to guide the LLM to answer all questions
    # in a single pass and format the output as a JSON object.
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

    # This chain defines the sequence of operations.
    # 1. The input `questions` list is joined into a single string for the retriever.
    # 2. The original `questions` list is passed through untouched for the prompt.
    # 3. The retriever fetches relevant context.
    # 4. The context and original questions are fed into the prompt.
    # 5. The formatted prompt is sent to the LLM.
    # 6. The LLM's output is parsed into the desired JSON structure.
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

# --- API Endpoint ---

@app.post("/hackrx/run", response_model=AnswerResponse)
def run_adjudication(
    request: QuestionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    API endpoint that processes a document to answer a list of questions in a single batch.
    """
    # Simple bearer token authentication. Ensure API_KEY is set in your .env file.
    api_key = os.getenv("API_KEY")
    if not api_key or authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key.")

    tmp_file_path = None
    try:
        # 1. Download and Load Document
        print(f"Downloading document from: {request.documents}")
        response = requests.get(str(request.documents))
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        
        # Use a temporary file to store the downloaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        print(f"Document saved to temporary file: {tmp_file_path}")

        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        
        # 2. Create In-Memory Vector Store
        print("Splitting documents and creating vector store...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        
        # Using a local, open-source embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # ChromaDB creates the vector store in memory
        vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})

        # 3. Initialize Chain and Process Questions in a Single Call
        qa_chain = get_qa_rag_chain(retriever)
        
        print(f"\nProcessing all {len(request.questions)} questions in a single batch...")
        # Invoke the chain ONCE with all questions.
        llm_response = qa_chain.invoke({"questions": request.questions})
        print(f"Generated Answers (batched): {llm_response}")
        
        # The final response is formatted according to the AnswerResponse model.
        return AnswerResponse(
            source_document=str(request.documents),
            answers=llm_response.get('answers', [])
        )

    except requests.exceptions.RequestException as e:
        print(f"Error downloading document: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download or access the document URL: {str(e)}")
    except Exception as e:
        # A general error handler for any other issues
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
    finally:
        # This block ensures the temporary file is always cleaned up
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "AI Document Q&A Processor API is running."}

# --- How to Run This App ---
#
# 1. Save this code as a Python file (e.g., `app.py`).
#
# 2. Create a `requirements.txt` file in the same directory with the following content:
#    fastapi
#    uvicorn[standard]
#    python-dotenv
#    langchain
#    langchain-google-genai
#    langchain-community
#    pypdf
#    chromadb
#    sentence-transformers
#    requests
#
# 3. Install the dependencies:
#    pip install -r requirements.txt
#
# 4. Create a `.env` file in the same directory with your API keys:
#    GOOGLE_API_KEY="your_google_api_key_here"
#    API_KEY="your_secret_bearer_token_here"
#
# 5. Run the server from your terminal:
#    python app.py
#
#    You should see output indicating the Uvicorn server has started on http://127.0.0.1:8000.

if __name__ == "__main__":
    # This block makes the script directly runnable.
    # By passing the `app` object directly to uvicorn.run, we avoid issues
    # where the script's filename doesn't match the string "main:app".
    # This is a more robust way to run the server for development.
    uvicorn.run(app, host="127.0.0.1", port=8000)

