Ishaan Lonial
CSE 256 - Final Project

# **The Huberman Lab - Fitness Q&A**

[**Access the Live Project on Streamlit**](https://cse256ishaanlonial.streamlit.app)
---
## **Overview**
This repository contains a fully deployed and functioning project: **The Huberman Lab - Fitness Q&A**, a **Retrieval-Augmented Generation (RAG)** system that combines advanced Natural Language Processing (NLP) techniques and OpenAI's GPT-4 model to deliver precise answers to health and fitness-related questions. The project extracts knowledge from Dr. Andrew Huberman's podcasts and uses a well-defined pipeline to answer user queries in a hyper-specific and domain-relevant manner.

This project is built to demonstrate not only technical proficiency in implementing a real-world NLP application but also the ability to fully deploy and host the system for public accessibility via **Streamlit**.

---
## **Features Implemented**
- **Dynamic Context Retrieval:** Utilizes TF-IDF-based similarity scoring to fetch the most relevant transcript chunks for a given query.
- **Domain-Specific Knowledge:** Answers are derived directly from Dr. Huberman's podcasts, ensuring responses are rich in scientific and technical accuracy.
- **Seamless Integration:** Built with OpenAI's GPT-4 model, leveraging its generative capabilities while grounding answers in the podcast dataset.
- **User-Friendly Interface:** A Streamlit-based interface makes it easy for users to interact with the model and ask questions in real-time.
- **Fully Deployed Application:** Hosted on Streamlit, accessible to anyone with an internet connection.

---
## **How It Works**
The pipeline for the project follows these steps:
1. **Data Collection:** Podcast videos are downloaded using `yt-dlp`, and audio files are extracted.
2. **Transcription:** Audio is transcribed into text using OpenAI's `Whisper` ASR model, retaining domain-specific terminology.
3. **Chunking & Preprocessing:** Transcripts are split into smaller chunks to meet GPT-4's token constraints, cleaned for irrelevant content, and indexed for retrieval using TF-IDF.
4. **Query Handling:** User queries are processed through TF-IDF, fetching the most relevant transcript chunks.
5. **Answer Generation:** Retrieved chunks are passed to GPT-4 with a carefully engineered prompt to generate domain-specific answers.
6. **Streamlit UI:** The entire system is integrated into a user-friendly Streamlit app for real-time Q&A.

---
## **Project Highlights**
1. **Fully Deployed Solution:** The system is live and functional at [cse256ishaanlonial.streamlit.app](https://cse256ishaanlonial.streamlit.app).
2. **Technical Sophistication:** Implements RAG techniques, leveraging TF-IDF and GPT-4 for seamless integration of retrieval and generation.
3. **User-Friendly Interaction:** Streamlit interface allows for intuitive real-time querying.

---
## **Challenges Addressed**
- **Token Management:** Successfully segmented transcripts into smaller chunks for compatibility with GPT-4.
- **Out-of-Context Queries:** Model gracefully falls back to GPT-4’s general knowledge for unrelated questions.
- **Deployment Issues:** Fully deployed the app on Streamlit, ensuring accessibility.

---
## **Future Directions**
- Dynamic chunking mechanisms for enhanced query understanding.
- Multi-modal support for richer outputs (audio, video, and text).
- Expansion to additional datasets and knowledge sources.

---
## **Getting Started**

### **Requirements**
To run the project locally, ensure you have:
- Python 3.8+
- Streamlit
- OpenAI's API key
- Whisper (for transcription)
- Required Python libraries (listed in `requirements.txt`)

### **Installation**
1. Clone the repository by running the command:
   `git clone https://github.com/your-repo/huberman-fitness-qna.git`
   Navigate to the project directory using `cd huberman-fitness-qna`.

2. Install dependencies by running:
   `pip install -r requirements.txt`

3. Set up the OpenAI API key as an environment variable by running:
   `export OPENAI_API_KEY=your_openai_api_key`

4. Run the Streamlit app by executing:
   `streamlit run app2.py`

---
## **Made by**
**Ishaan Lonial**
Email: ilonial@ucsd.edu