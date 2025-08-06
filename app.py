import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Setup prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Hey, you are a helpful assistant. Please respond to the question asked"),
        ("user", "Question:{question}")
    ]
)

## LLM setup
llm = Ollama(model="llama3.2:1b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

## Streamlit framework
st.set_page_config(page_title="LLAMA3 Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        height: 45px;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 8px;
        font-size: 16px;
        height: 45px;
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# UI
st.title("🦙 LangChain + LLAMA3 Chatbot")
st.markdown("Ask anything and get a smart answer!")

with st.container():
    input_text = st.text_input("💬 Enter your question here")

    if input_text:
        with st.spinner("Thinking... 🤔"):
            response = chain.invoke({"question": input_text})
        st.success("✅ Answer:")
        st.markdown(f"**{response}**")