# 🤖 AskAnkon - AI Chatbot Powered by Gemini, Groq & FAISS

Welcome to **AskAnkon**, an AI-powered chatbot that knows everything about me — because I custom-trained it with my personal data! Built using **Gemini**, **Groq API**, and a **FAISS** vector database, this chatbot can respond intelligently to any queries about me.

---

## 🧠 What is AskAnkon?

**AskAnkon** is a personalized AI chatbot that uses powerful LLMs and vector search to provide accurate and context-aware answers about me. It is trained using my custom data, allowing it to serve as an AI version of myself.

---

## 🚀 Features

- 🔍 **Context-aware responses** about my background, skills, projects, and more.
- 🧠 **Custom-trained embeddings** using my personal documents and profile data.
- ⚡ **Fast and lightweight** inference using the **Groq API** for ultra-fast response times.
- 🗂️ Integrated **FAISS** vector store for efficient similarity search.
- 🔐 Secure API handling for Gemini and Groq.

---

## 🛠️ Tech Stack

| Tech        | Usage                                |
|-------------|--------------------------------------|
| **Gemini**  | LLM for response generation          |
| **Groq API**| High-speed model inference           |
| **FAISS**   | Vector similarity search database    |
| **Python**  | Backend logic                        |

---

## 🧩 How It Works

1. **Data Ingestion**  
   Your personal documents, CV, and other resources are chunked and embedded.

2. **FAISS Indexing**  
   The embeddings are stored in a FAISS vector database for efficient retrieval.

3. **Query Handling**  
   User input is converted into a query vector and matched against the stored vectors in the database.

4. **LLM Response**  
   Gemini or Groq API generates a human-like response using the relevant retrieved information.

5. **Output**  
   A natural language answer is returned to the user based on the context and intent of the query.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

