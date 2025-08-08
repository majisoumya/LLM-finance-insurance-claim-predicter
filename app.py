import os
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# Load environment variables from a .env file
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Document Q&A", layout="wide")

# --- Helper Functions ---
def load_css(file_path):
    """Loads a CSS file for styling the app."""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_lottieurl(url: str):
    """Fetches a Lottie JSON from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Load CSS for glassmorphism effect ---
# Creates a dummy style.css if it doesn't exist
if not os.path.exists("style.css"):
    with open("style.css", "w") as f:
        f.write("""
        .glass-container {
            background: rgba(40, 43, 54, 0.6); backdrop-filter: blur(10px);
            border-radius: 15px; padding: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(5px);
            border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .centered-title { text-align: center; }
        """)
load_css("style.css")

# --- Main UI Layout ---
st.markdown("<div class='glass-container'>", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='centered-title'>AI Document Q&A Processor</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    lottie_json = load_lottieurl("https://lottie.host/e2c0469b-2a3b-4171-843c-68b3684a8673/h7Vd1Yv7fM.json")
    if lottie_json:
        st_lottie(lottie_json, speed=1, height=150, key="robot")

# API and Document Inputs
st.markdown("<h3>⚙️ Configuration</h3>", unsafe_allow_html=True)
# The FastAPI endpoint URL. Default is for local development.
api_endpoint = st.text_input("Backend API Endpoint:", "http://127.0.0.1:8000/hackrx/run")
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
    # Input validation
    if not all([api_endpoint, api_key, document_url, questions_text.strip()]):
        st.warning("⚠️ Please fill in all fields: API Endpoint, API Key, Document URL, and Questions.")
    else:
        questions_list = [q.strip() for q in questions_text.split("\n") if q.strip()]
        
        # Prepare the request payload and headers
        payload = {
            "documents": document_url,
            "questions": questions_list
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            with st.spinner("Sending request to backend... Please wait, this may take a moment."):
                # Make the API call to the FastAPI backend
                response = requests.post(api_endpoint, json=payload, headers=headers)
                
                # Check for successful response
                if response.status_code == 200:
                    results = response.json()
                    answers = results.get("answers")

                    # Check if the backend returned any answers
                    if answers:
                        st.success("✅ All questions processed successfully!")
                        st.markdown("<h3>📝 Final Response</h3>", unsafe_allow_html=True)
                        # Display each answer in its own glass box
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
                    else:
                        # Handle case where API succeeded but returned no answers
                        st.info("ℹ️ The request was successful, but the AI did not return any answers for the given questions.")
                else:
                    # Display error from the API
                    st.error(f"❌ API Error (Status {response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Network Error: Could not connect to the backend API at {api_endpoint}. Please ensure it is running. Details: {e}")

# --- Glass Container End ---
st.markdown("</div>", unsafe_allow_html=True)
