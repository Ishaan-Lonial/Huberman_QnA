import streamlit as st
import pandas as pd
from langchain.schema import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import os

# Get the OpenAI API key securely
OpenAI_key = os.getenv("OPENAI_API_KEY")  # Environment variable for OpenAI API Key

# Ensure OpenAI_key is set
if not OpenAI_key:
    st.error("OpenAI API key is missing. Set it as an environment variable before running the app.")
    st.stop()

# Load the CSV file directly from the repository
csv_file = "video_transcripts_chunks.csv"  # Ensure the file is in your repository's root

if not os.path.exists(csv_file):
    st.error(f"File {csv_file} not found. Please ensure it exists in the correct directory.")
    st.stop()

# Read the CSV file
df = pd.read_csv(csv_file)

# Convert CSV data to Document objects
documents = [
    Document(page_content=row['transcripts'], metadata={'title': row['videotitle']})
    for index, row in df.iterrows()
]

# Extract texts and titles from documents
texts = [doc.page_content for doc in documents]
titles = [doc.metadata['title'] for doc in documents]

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=OpenAI_key)

# Define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a fitness expert assistant."),
        ("system", "Examples:\n"
                   "Q: How can I improve my breathing?\n"
                   "A: Practice diaphragmatic breathing and incorporate box breathing during workouts.\n"
                   "\n"
                   "Q: What are the best ways to recover after a workout?\n"
                   "A: Use active recovery techniques like stretching and foam rolling."),
        ("system", "Now answer the user's query using the provided transcripts: {context}."),
        ("human", "{question}")
    ]
)

# Subchain for generating an answer
answer_chain = LLMChain(llm=llm, prompt=prompt)

# Function to filter and format relevant documents
def get_relevant_docs(query: str, texts: List[str], titles: List[str], top_k: int = 5) -> str:
    """Retrieve and format the top-k relevant documents based on cosine similarity."""
    vectorizer = TfidfVectorizer().fit_transform(texts + [query])
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    top_indices = cosine_sim.argsort()[-top_k:][::-1]

    # Format the top documents
    formatted_docs = [
        f"Video Title: {titles[idx]}\nTranscript: {texts[idx]}"
        for idx in top_indices
    ]
    return "\n\n".join(formatted_docs)

# Streamlit Interface
st.title("The Huberman Lab - Fitness Q&A")

# Input for user's question
question = st.text_input("Ask your Question:")

if st.button("Get Answer"):
    if question:
        # Get relevant documents
        context = get_relevant_docs(question, texts, titles)

        # Run the LLM chain
        result = answer_chain.run({"question": question, "context": context})

        # Display the result
        st.write("### Question")
        st.write(question)
        st.write("### Answer")
        st.write(result)
    else:
        st.write("Please enter a question.")
