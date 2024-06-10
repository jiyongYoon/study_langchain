from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI


def generate_gemini_pro_llm(temperature=0):
    load_dotenv()

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=temperature
    )

    return llm
