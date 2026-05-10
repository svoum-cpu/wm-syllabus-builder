import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from generate_policy import PROMPT_TEMPLATE 

# 1. Page Config
st.set_page_config(page_title="W&M AI Policy Architect", page_icon="🛡️", layout="centered")

# --- INITIALIZE SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []
if "last_policy" not in st.session_state:
    st.session_state.last_policy = None

# 2. W&M Brand Identity & Styling
WM_GREEN = "#115740"
WM_GOLD = "#B9975B"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    .stApp {{ background-color: #FFFFFF; }}
    .hero {{
        background: {WM_GREEN}; padding: 4rem 2rem; border-radius: 15px; color: white;
        text-align: center; margin-bottom: 3rem; border-bottom: 8px solid {WM_GOLD};
    }}
    .hero h1 {{ font-family: 'Libre Baskerville', serif; font-size: 3rem; color: white; }}
    .section-header {{
        font-family: 'Libre Baskerville', serif; color: {WM_GREEN};
        border-bottom: 2px solid #EEE; padding-bottom: 10px; margin-top: 2rem; margin-bottom: 1.5rem;
    }}
    .policy-memo {{
        background-color: #fcfcfc; padding: 3rem; border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #EEE;
        border-top: 12px solid {WM_GOLD}; margin-top: 2rem;
    }}
    .stButton>button {{
        width: 100%; height: 4rem; background-color: {WM_GREEN}; color: white !important;
        font-size: 1.2rem; font-weight: bold; border: 2px solid {WM_GOLD}; border-radius: 50px;
    }}
    .stButton>button:hover {{ background-color: {WM_GOLD}; color: {WM_GREEN} !important; }}
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📜 Session History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.last_policy = None
        st.rerun()

    if not st.session_state.history:
        st.info("No policies generated yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            # Clean up the display text for the sidebar
            with st.expander(f"{item['time']} - {item['name']}"):
                st.write(item['content'][:150] + "...")
                st.download_button("Download", item['content'], file_name=f"WM_{item['name']}.txt", key=f"sidebar_dl_{idx}")

# --- HERO ---
st.markdown(f'<div class="hero"><h1>🛡️ AI Policy Architect</h1><p>School of Computing, Data Sciences & Physics</p></div>', unsafe_allow_html=True)

# --- 1. PERSISTENT REVEAL (Shows the memo if one was just created) ---
if st.session_state.last_policy:
    st.markdown('<h2 class="section-header">🏛️ Generated Policy Memo</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="policy-memo">', unsafe_allow_html=True)
    st.markdown(st.session_state.last_policy)
    st.markdown('</div>', unsafe_allow_html=True)
    st.download_button("💾 Download .txt for Syllabus", st.session_state.last_policy, file_name="WM_Syllabus_Policy.txt")

# --- 2. INPUT SECTION ---
st.markdown('<h2 class="section-header">📝 Assignment Context</h2>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    class_name = st.text_input("Course Number (e.g., DATA 305)")
with c2:
    weight = st.text_input("Assignment Weight (%)")

assignment_name = st.text_input("Assignment Title")
assignment_details = st.text_area("Detailed Prompt / Task Description", height=200)
learning_objs = st.text_area("Learning Objectives & Grading Criteria", height=100)

# --- 3. THE BUTTON & AI LOGIC ---
if st.button("Create Official Policy Memo"):
    if not class_name or not assignment_details:
        st.error("Please provide at least the Course Number and Assignment Details.")
    else:
        with st.spinner("⚖️ Consulting William & Mary Academic Standards..."):
            # Backend RAG Logic
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
            search_query = f"{class_name} and William & Mary Honor Code"
            results = db.similarity_search(search_query, k=5)
            context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
            
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(
                context=context_text, class_name=class_name,
                assignment_name=assignment_name, assignment_details=assignment_details,
                weight=weight, learning_objectives=learning_objs
            )

            model = ChatGoogleGenerativeAI(model="gemini-flash-latest")
            response = model.invoke(prompt)
            
            # --- STRING CLEANING (The fix for your sidebar error) ---
            if isinstance(response.content, list):
                raw_text = response.content[0].get('text', str(response.content))
            else:
                raw_text = response.content
            
            clean_text = str(raw_text).strip()

            # --- SAVE TO SESSION STATE ---
            st.session_state.last_policy = clean_text
            st.session_state.history.append({
                "name": assignment_name if assignment_name else class_name,
                "content": clean_text,
                "time": datetime.now().strftime("%I:%M %p")
            })
            
            st.rerun()