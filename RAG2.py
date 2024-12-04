import pandas as pd
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set the OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")  # Replace with your actual API key

# Load the CSV file
csv_file = 'video_transcripts_chunks.csv'  # Ensure this file path is correct
df = pd.read_csv(csv_file)

# Convert CSV data to Document objects
documents = [
    Document(page_content=row['transcripts'], metadata={'title': row['videotitle']})
    for index, row in df.iterrows()
]

# Convert documents to a list of texts and titles
texts = [doc.page_content for doc in documents]
titles = [doc.metadata['title'] for doc in documents]

# Initialize the OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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


# Initialize LLM
from langchain_community.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=OPENAI_API_KEY)

# Subchain for generating an answer
answer_chain = LLMChain(llm=llm, prompt=prompt)

# Function to filter and format relevant documents
def get_relevant_docs(query: str, texts: list, titles: list, top_k: int = 5) -> str:
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

# Example usage
question = "How do I lose weight?"

# Get the relevant documents
context = get_relevant_docs(question, texts, titles)

# Run the LLM chain
result = answer_chain.run({"question": question, "context": context})

print(f"Question: {question}")
print("Answer:", result)