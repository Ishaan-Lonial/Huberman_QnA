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

# Set the OpenAI API Key
OpenAI_key = os.environ.get(
    "OPENAI_API_KEY",
    "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"
)

# Ensure OpenAI_key is set
if not OpenAI_key:
    raise ValueError("OpenAI API key is missing. Set it in the OPENAI_API_KEY environment variable or directly in the code.")

# Streamlit Interface
st.title("The Huberman Lab - Fitness Q&A")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file:", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

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

    question = st.text_input("Ask your Question:")

    if st.button("Get Answer"):
        if question:
            # Get relevant documents
            context = get_relevant_docs(question, texts, titles)

            # Run the LLM chain
            result = answer_chain.run({"question": question, "context": context})

            st.write("### Question")
            st.write(question)
            st.write("### Answer")
            st.write(result)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a CSV file to proceed.")