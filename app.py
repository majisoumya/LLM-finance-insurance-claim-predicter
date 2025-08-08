import os
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import time

# Load environment variables from a .env file
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="🤖",
    layout="wide"
)

# --- Helper Functions ---
def load_css(file_path):
    """Loads a CSS file for styling the app."""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottieurl(url: str):
    """Fetches a Lottie JSON from a URL with error handling."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error loading Lottie animation: {e}")
        return None

def display_results(answers: list):
    """A helper function to display the Q&A results in a consistent format."""
    st.success("🎉 All questions processed successfully!")
    st.markdown("<h3>📝 Final Response</h3>", unsafe_allow_html=True)
    for i, item in enumerate(answers):
        question = item.get("question", "N/A")
        answer = item.get("answer", "N/A")
        st.markdown(f"""
        <div class="glass-card">
            <p style="color: #A5B4FC; font-weight: bold;">Question {i+1}: {question}</p>
            <hr style="border-color: rgba(255, 255, 255, 0.2);">
            <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)

# --- Load CSS for styling ---
if not os.path.exists("style.css"):
    with open("style.css", "w") as f:
        f.write("""
        .glass-container {
            background: rgba(40, 43, 54, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .centered-title {
            text-align: center;
        }
        """)
load_css("style.css")

# --- Main UI Layout ---
st.markdown("<div class='glass-container'>", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='centered-title'>AI Document Q&A Processor</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    lottie_json = load_lottieurl("https://lottie.host/e2c0469b-2a3b-4171-843c-68b3684a8673/h7Vd1Yv7fM.json")
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=150, key="robot")

# API and Document Inputs Section
st.markdown("<h3>⚙️ Configuration</h3>", unsafe_allow_html=True)
api_endpoint = st.text_input("Backend API Endpoint:", "http://127.0.0.1:8000")
api_key = st.text_input("API Key:", type="password", help="Your secret API key for the backend service.")

st.markdown("<h3>📄 Document & Questions</h3>", unsafe_allow_html=True)
document_url = st.text_input(
    "Enter the URL of the policy PDF document:",
    "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
)
default_questions = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses?",
]
questions_text = st.text_area(
    "Enter your questions, one per line:",
    "\n".join(default_questions),
    height=150
)

# --- Processing Logic ---
if st.button("🚀 Process Document and Questions"):
    if not all([api_endpoint, api_key, document_url, questions_text.strip()]):
        st.warning("⚠️ Please fill in all fields: API Endpoint, API Key, Document URL, and Questions.")
    else:
        questions_list = [q.strip() for q in questions_text.split("\n") if q.strip()]
        
        payload = {
            "documents": document_url,
            "questions": questions_list
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            with st.spinner("🚀 Sending request to backend..."):
                # Use the synchronous endpoint for simplicity now
                response = requests.post(f"{api_endpoint}/hackrx/run_sync", json=payload, headers=headers)
                
                # --- FIX: Handle both 200 (direct response) and 202 (polling) ---

                # Case 1: The job finished quickly and returned the full result
                if response.status_code == 200:
                    results = response.json()
                    answers = results.get("answers", [])
                    display_results(answers)

                # Case 2: The job will take time, so we need to poll
                elif response.status_code == 202:
                    job_id = response.json().get("job_id")
                    st.info(f"✅ Request accepted! Your job ID is: {job_id}. Now polling for results...")

                    while True:
                        with st.spinner("⏳ Checking job status... Please wait."):
                            status_response = requests.get(f"{api_endpoint}/status/{job_id}")
                            
                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                if status_data["status"] == "complete":
                                    answers = status_data.get("result", {}).get("answers", [])
                                    display_results(answers)
                                    break
                                elif status_data["status"] == "failed":
                                    st.error("❌ The job failed to process on the backend.")
                                    break
                            else:
                                st.error(f"❌ Error checking status (Status {status_response.status_code}): {status_response.text}")
                                break
                        time.sleep(5) 
                
                # Case 3: An actual error occurred
                else:
                    st.error(f"❌ API Error (Status {response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network Error: Could not connect to the backend API at {api_endpoint}. Please ensure it is running. Details: {e}")

st.markdown("</div>", unsafe_allow_html=True)
