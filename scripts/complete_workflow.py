import sys
import os
import json
import pdfplumber
import numpy as np
import faiss
from collections import deque

# Add the parent directory to the Python path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.groq_api_config import GroqAPI
from config.settings import GROQ_API_KEY

# def extract_text_from_pdf_via_groq(pdf_path):
#     """Extracts text from a PDF using an external tool and processes it with Groq API."""
    
#     # Ideally, you'd first extract text from the PDF using a method like pdfplumber or another PDF tool
#     # For this example, let's assume text is extracted as a string.
#     text = ""
    
#     # Use your preferred method to extract text from the PDF (this could be an external service or library)
#     try:
#         with open(pdf_path, 'rb') as f:
#             # For now, you can pass the file content to Groq API for analysis
#             groq_api = GroqAPI(api_key=GROQ_API_KEY)
#             messages = [{"role": "user", "content": f"Extract and process the following PDF content: {f.read()}"}]
#             response = groq_api.get_chat_completion(messages)
#             text = response['choices'][0]['message']['content']
    
#     except Exception as e:
#         print(f"Error processing {pdf_path} with Groq API: {e}")
    
#     return text




def extract_text_from_pdf_via_groq(pdf_path):
    """Extracts text from a PDF using pdfplumber and processes it with Groq API."""
    
    # Initialize text variable to hold extracted text
    text = ""
    
    try:
        # Use pdfplumber to extract text from the PDF
        with pdfplumber.open(pdf_path) as pdf:
            # Extract text from each page and append to the 'text' variable
            for page in pdf.pages:
                text += page.extract_text() or ""  # Use empty string if page.extract_text() returns None
        
        if not text.strip():
            raise ValueError("No text extracted from the PDF.")
        
        # Pass extracted text to the Groq API for further processing
        groq_api = GroqAPI(api_key=GROQ_API_KEY)
        messages = [{"role": "user", "content": f"Process the following extracted PDF content: {text}"}]
        
        response = groq_api.get_chat_completion(messages)
        
        # Extract processed content from the response
        text = response
    
    except Exception as e:
        print(f"Error processing {pdf_path} with Groq API: {e}")
    
    return text



def clean_text(text):
    """Cleans the extracted text."""
    if not text:
        return ""
    
    # Basic cleaning
    cleaned_text = text.strip()
    # Remove lines that are too short (likely headers/footers)
    cleaned_text = "\n".join([line for line in cleaned_text.split("\n") if len(line.strip()) > 3])
    return cleaned_text

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks for better embedding generation."""
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def generate_embeddings_for_chunks(chunks):
    """Generates embeddings for text chunks using Groq API."""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in environment variables.")
        return []
    
    groq_api = GroqAPI(api_key=GROQ_API_KEY)
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        try:
            print(f"Generating embedding for chunk {i+1}/{len(chunks)}...")
            
            # Use chat completion to generate a summary/representation of the chunk
            messages = [{"role": "user", "content": f"Summarize this text in 2-3 sentences: {chunk[:500]}..."}]
            response = groq_api.get_chat_completion(messages)
            
            # Create a simple embedding representation (you might want to use a proper embedding model)
            embedding_data = {
                "text": chunk,
                "summary": response,
                "embedding": [ord(c) % 100 for c in response[:100]]  # Simple hash-based embedding
            }
            
            embeddings.append(embedding_data)
            
        except Exception as e:
            print(f"Error generating embedding for chunk {i+1}: {e}")
            continue
    
    return embeddings

def save_embeddings(embeddings, output_dir):
    """Saves embeddings to individual JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, embedding in enumerate(embeddings):
        filename = f"embedding_{i:03d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(embedding, f, indent=2, ensure_ascii=False)
        
        print(f"Saved embedding to {filepath}")

def create_faiss_index(embeddings):
    """Creates a FAISS index from the provided embeddings."""
    if not embeddings:
        raise ValueError("No embeddings found.")
    
    # Convert embeddings list into a NumPy array
    embeddings_array = np.array([embedding['embedding'] for embedding in embeddings]).astype('float32')
    
    # Create the FAISS index
    index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 distance metric
    
    # Add embeddings to the FAISS index
    index.add(embeddings_array)
    
    return index

def save_faiss_index(index, index_path):
    """Saves the FAISS index to a file."""
    faiss.write_index(index, index_path)

def load_faiss_index(index_path):
    """Loads the FAISS index from a file."""
    if not os.path.exists(index_path):
        print(f"FAISS index not found at {index_path}")
        return None
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    return index

def query_faiss_index(query, index, top_k=9):
    """Queries the FAISS index for the most relevant chunks."""
    if index is None:
        print("No FAISS index available for querying.")
        return None
    
    # Generate a simple embedding for the query (using the same method as in generate_embeddings_for_chunks)
    groq_api = GroqAPI(api_key=GROQ_API_KEY)
    messages = [{"role": "user", "content": f"Summarize this query in 2-3 sentences: {query}"}]
    response = groq_api.get_chat_completion(messages)
    
    # Create the same type of embedding as used in the index
    query_embedding = [ord(c) % 100 for c in response[:100]]
    
    # Convert query embedding to a NumPy array (FAISS expects float32 type)
    query_embedding_array = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    # Perform the search on FAISS index
    _, indices = index.search(query_embedding_array, top_k)
    
    return indices

def generate_response_with_context(query, indices, embeddings_dir, memory):
    """Generate a response using the retrieved chunks."""
    context = ""

    # Load the top-k chunks from the embeddings directory based on indices
    print(f"Indices retrieved: {indices}")  # Debug print to check indices

    for idx in indices[0]:
        embedding_file = os.path.join(embeddings_dir, f"embedding_{idx:03d}.json")
        print(f"Trying to load embedding file: {embedding_file}")  # Debug print to check file path
        
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                embedding_data = json.load(f)
                print(f"Embedding data: {embedding_data}")  # Debug print to check the content of the file
                context += embedding_data['summary'] + "\n"  # You can adjust what to append (summary, text, etc.)
        except FileNotFoundError:
            context += "Sorry, I couldn't find relevant context for this query.\n"
    
    # If no context was loaded, prepare the fallback response
    if not context.strip():
        context = "Currently, I don't have an exact answer. You can contact me live through Email: ankontheanalyst@gmail.com or Phone: +8801676-597334. My name is Arafat Hossain Ankon."
    
    # Add previous messages to memory (up to the last 5)
    memory.append(f"User: {query}")
    if len(memory) > 5:
        memory.pop(0)
    
    context = "\n".join(memory) + "\n" + context
    
    # Add a personalized greeting at the start of the response
    context = "Hi, I am Arafat Hossain Ankon. " + context
    
    # Generate response based on the context using Groq API
    groq_api = GroqAPI(api_key=GROQ_API_KEY)
    messages = [{"role": "user", "content": f"Answer based on the following context: {context}\n{query}"}]
    response = groq_api.get_chat_completion(messages)
    
    return response


def main():
    """Main workflow function."""
    print("Starting complete workflow...")
    
    # Initialize memory (store last 5 messages)
    memory = deque(maxlen=5)
    
    # Step 1: Extract text from PDF
    pdf_path = "data/raw_data/t2.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at {pdf_path}")
        return
    
    print("Step 1: Extracting text from PDF...")
    extracted_text = extract_text_from_pdf_via_groq(pdf_path)
    
    if not extracted_text:
        print("No text extracted from PDF.")
        return
    
    # Step 2: Clean the text
    print("Step 2: Cleaning text...")
    cleaned_text = clean_text(extracted_text)
    
    # Step 3: Split into chunks
    print("Step 3: Splitting text into chunks...")
    chunks = split_text_into_chunks(cleaned_text)
    print(f"Created {len(chunks)} text chunks")
    
    # Step 4: Generate embeddings
    print("Step 4: Generating embeddings...")
    embeddings = generate_embeddings_for_chunks(chunks)
    
    if not embeddings:
        print("No embeddings generated.")
        return
    
    # Step 5: Save embeddings
    print("Step 5: Saving embeddings...")
    embeddings_dir = "data/embeddings"
    save_embeddings(embeddings, embeddings_dir)
    
    # Step 6: Create FAISS index
    print("Step 6: Creating FAISS index...")
    faiss_index_path = 'data/embeddings/faiss_index.index'
    index = create_faiss_index(embeddings)
    save_faiss_index(index, faiss_index_path)
    print(f"FAISS index created and saved to {faiss_index_path}")
    
    # Step 7: Start conversational loop
    print("\nStep 7: Starting conversational interface...")
    print("You can now ask questions about the document. Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nAsk a question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a question.")
                continue
            
            print("Searching for relevant information...")
            
            # Query the FAISS index
            indices = query_faiss_index(query, index, top_k=5)
            if indices is not None:
                print(f"Found {len(indices[0])} relevant chunks.")
                
                # Generate response using the relevant context and memory
                response = generate_response_with_context(query, indices, embeddings_dir, memory)
                print("\nAnswer:", response)
                print("-" * 50)
            else:
                print("Could not find relevant information for your query.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing your question: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()











# Gemini API



# import sys
# import os
# import json
# import requests
# import numpy as np
# import faiss
# import base64
# from collections import deque
# from dotenv import load_dotenv  # To load environment variables from .env file

# # Add the parent directory to the Python path so we can import from config
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Load environment variables from .env file
# load_dotenv()

# # Get GEMINI API Key from the environment variable
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure the API key is correctly set in your .env file

# import os
# import base64
# import requests

# import os
# import base64
# import requests

# def extract_text_from_pdf_via_gemini(pdf_path, output_dir="data", max_tokens=16384):
#     """Extracts text from a PDF using the Gemini API and stores it locally in a text file.
#     This function preserves the text exactly as it appears in the PDF, word by word, without modification."""
    
#     text = ""
#     try:
#         with open(pdf_path, 'rb') as f:
#             pdf_data = f.read()

#         # Convert PDF to base64 for Gemini API
#         pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

#         url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
#         # Request payload for Gemini API
#         payload = {
#             "contents": [{
#                 "parts": [{
#                     "inline_data": {
#                         "mime_type": "application/pdf",
#                         "data": pdf_base64
#                     }
#                 }]}
#             ],
#             "generationConfig": {
#                 "temperature": 0.1,
#                 "topK": 1,
#                 "topP": 1,
#                 "maxOutputTokens": max_tokens,  # Use the increased token limit
#             }
#         }

#         # Send the request to Gemini API
#         response = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload)

#         # Check if the request was successful
#         if response.status_code == 200:
#             result = response.json()
#             if 'candidates' in result and len(result['candidates']) > 0:
#                 text = result['candidates'][0]['content']['parts'][0]['text']
#             else:
#                 print("No text extracted from PDF")
#         else:
#             print(f"Error extracting text with Gemini API: {response.status_code}, {response.text}")
        
#     except Exception as e:
#         print(f"Error processing {pdf_path} with Gemini API: {e}")
    
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Automatically generate the output file name and store the extracted text
#     if text:
#         # Generate a unique file name by appending the number to avoid overwriting
#         base_filename = f"extracted_text_{os.path.basename(pdf_path)}"
#         output_file_path = os.path.join(output_dir, f"{base_filename}.txt")

#         # Check if file already exists and increment the filename if needed
#         count = 1
#         while os.path.exists(output_file_path):
#             output_file_path = os.path.join(output_dir, f"{base_filename}_{count}.txt")
#             count += 1

#         # Save the extracted text to the file
#         with open(output_file_path, 'w', encoding='utf-8') as output_file:
#             output_file.write(text)
#         print(f"Extracted text saved to {output_file_path}")
#     else:
#         print("No text extracted to save.")

#     return text



# def clean_text(text):
#     """Cleans the extracted text."""
#     if not text:
#         return ""
    
#     cleaned_text = text.strip()
#     cleaned_text = "\n".join([line for line in cleaned_text.split("\n") if len(line.strip()) > 3])
#     return cleaned_text

# def split_text_into_chunks(text, chunk_size=1000, overlap=200):
#     """Splits text into overlapping chunks for better embedding generation."""
#     if not text:
#         return []
    
#     chunks = []
#     start = 0
    
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
        
#         if chunk.strip():
#             chunks.append(chunk.strip())
        
#         start = end - overlap
#         if start >= len(text):
#             break
    
#     return chunks



# def generate_embeddings_for_chunks(chunks, embedding_dim=100, max_tokens=8192):
#     """Generates embeddings for text chunks using the Gemini API.
#     This function processes each chunk as-is and stores the embeddings."""
    
#     if not GEMINI_API_KEY:
#         print("Error: GEMINI_API_KEY not found in environment variables.")
#         return []
    
#     embeddings = []
    
#     for i, chunk in enumerate(chunks):
#         try:
#             print(f"Generating embedding for chunk {i+1}/{len(chunks)}...")

#             # Send request to Gemini API for text processing (embedding generation)
#             url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            
#             # Split large chunks into smaller ones if they exceed token limit (max_tokens)
#             if len(chunk) > max_tokens:
#                 chunk = chunk[:max_tokens]  # Truncate the chunk to fit the max token limit

#             payload = {
#                 "contents": [{
#                     "parts": [{
#                         "text": f"Don't Summarize : {chunk[:500]}..."
#                     }]
#                 }],
#                 "generationConfig": {
#                     "temperature": 0.3,
#                     "topK": 1,
#                     "topP": 1,
#                     "maxOutputTokens": 2000,  # Increased to allow larger responses
#                 }
#             }

#             # Send the request to Gemini API
#             response = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload)

#             if response.status_code == 200:
#                 result = response.json()
#                 if 'candidates' in result and len(result['candidates']) > 0:
#                     summary = result['candidates'][0]['content']['parts'][0]['text']
#                 else:
#                     summary = chunk[:100]  # Fallback to first 100 chars
#             else:
#                 print(f"Error generating embedding for chunk {i+1}: {response.status_code}, {response.text}")
#                 summary = chunk[:100]  # Fallback

#             # Create a simple hash-based embedding with fixed length
#             embedding_vector = [ord(c) % 100 for c in summary[:100]]
#             # Pad or truncate to ensure consistent length of 100
#             if len(embedding_vector) < 100:
#                 embedding_vector.extend([0] * (100 - len(embedding_vector)))
#             else:
#                 embedding_vector = embedding_vector[:100]
            
#             # Store embedding data for each chunk
#             embedding_data = {
#                 "text": chunk,
#                 "summary": summary,
#                 "embedding": embedding_vector
#             }
            
#             embeddings.append(embedding_data)
            
#         except Exception as e:
#             print(f"Error generating embedding for chunk {i+1}: {e}")
#             continue
    
#     return embeddings


# def save_embeddings(embeddings, output_dir):
#     """Saves embeddings to individual JSON files."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     for i, embedding in enumerate(embeddings):
#         filename = f"embedding_{i:03d}.json"
#         filepath = os.path.join(output_dir, filename)
        
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(embedding, f, indent=2, ensure_ascii=False)
        
#         print(f"Saved embedding to {filepath}")

# def create_faiss_index(embeddings):
#     """Creates a FAISS index from the provided embeddings."""
#     if not embeddings:
#         raise ValueError("No embeddings found.")
    
#     embeddings_array = np.array([embedding['embedding'] for embedding in embeddings]).astype('float32')
    
#     # Create the FAISS index
#     index = faiss.IndexFlatL2(embeddings_array.shape[1])  # L2 distance metric
#     index.add(embeddings_array)
    
#     return index

# def save_faiss_index(index, index_path):
#     """Saves the FAISS index to a file."""
#     faiss.write_index(index, index_path)

# def load_faiss_index(index_path):
#     """Loads the FAISS index from a file."""
#     if not os.path.exists(index_path):
#         print(f"FAISS index not found at {index_path}")
#         return None
#     print(f"Loading FAISS index from {index_path}...")
#     index = faiss.read_index(index_path)
#     return index

# def query_faiss_index(query, index, top_k=3):
#     """Queries the FAISS index for the most relevant chunks."""
#     if index is None:
#         print("No FAISS index available for querying.")
#         return None
    
#     # Generate a simple embedding for the query (using the same method as in generate_embeddings_for_chunks)
#     if not GEMINI_API_KEY:
#         print("Error: GEMINI_API_KEY not found in environment variables.")
#         return None
    
#     try:
#         # Send request to Gemini API for query summarization
#         url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
#         payload = {
#             "contents": [{
#                 "parts": [{
#                     "text": f"Summarize this query in 2-3 sentences: {query}"
#                 }]
#             }],
#             "generationConfig": {
#                 "temperature": 0.3,
#                 "topK": 1,
#                 "topP": 1,
#                 "maxOutputTokens": 2000,
#             }
#         }
        
#         response = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload)
        
#         if response.status_code == 200:
#             result = response.json()
#             if 'candidates' in result and len(result['candidates']) > 0:
#                 summary = result['candidates'][0]['content']['parts'][0]['text']
#             else:
#                 summary = query[:100]  # Fallback
#         else:
#             print(f"Error generating query embedding: {response.status_code}, {response.text}")
#             summary = query[:100]  # Fallback
        
#         # Create the same type of embedding as used in the index
#         query_embedding = [ord(c) % 100 for c in summary[:100]]
#         # Pad or truncate to ensure consistent length of 100
#         if len(query_embedding) < 100:
#             query_embedding.extend([0] * (100 - len(query_embedding)))
#         else:
#             query_embedding = query_embedding[:100]
        
#         # Convert query embedding to a NumPy array (FAISS expects float32 type)
#         query_embedding_array = np.array(query_embedding).astype('float32').reshape(1, -1)
        
#         # Perform the search on FAISS index
#         _, indices = index.search(query_embedding_array, top_k)
        
#         return indices
        
#     except Exception as e:
#         print(f"Error in query_faiss_index: {e}")
#         return None

# import requests
# import json
# import os

# def generate_response_with_context(query, indices, embeddings_dir, memory):
#     """Generate a response using the retrieved chunks, ensuring it is at least 5-6 lines long."""
#     context = ""

#     # Load the top-k chunks from the embeddings directory based on indices
#     for idx in indices[0]:
#         embedding_file = os.path.join(embeddings_dir, f"embedding_{idx:03d}.json")
#         try:
#             with open(embedding_file, 'r', encoding='utf-8') as f:
#                 embedding_data = json.load(f)
#                 context += embedding_data['summary'] + "\n"  # You can adjust what to append (summary, text, etc.)
#         except FileNotFoundError:
#             context += "Sorry, I couldn't find relevant context for this query.\n"
    
#     # If no context was loaded, prepare the fallback response
#     if not context.strip():
#         context = "Currently, I don't have an exact answer. You can contact me live through Email: ankontheanalyst@gmail.com or Phone: +8801676-597334. My name is Arafat Hossain Ankon."
    
#     # Add previous messages to memory (up to the last 5)
#     memory.append(f"User: {query}")
#     if len(memory) > 5:
#         memory.pop(0)
    
#     context = "\n".join(memory) + "\n" + context
    
#     # Add a personalized greeting at the start of the response
#     context = "Hi, I am Arafat Hossain Ankon. " + context
    
#     # Generate response based on the context using Gemini API
#     url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
#     payload = {
#         "contents": [{
#             "parts": [{
#                 "text": f"Please provide a detailed, 5-6 line answer based on the following context: {context}\n\nQuestion: {query}"
#             }]
#         }],
#         "generationConfig": {
#             "temperature": 0.7,       # Increased to make the response more dynamic
#             "topK": 1,
#             "topP": 1,
#             "maxOutputTokens": 1000,  # Increased token limit to generate longer responses
#         }
#     }
    
#     response = requests.post(f"{url}?key={GEMINI_API_KEY}", json=payload)
    
#     if response.status_code == 200:
#         result = response.json()
#         if 'candidates' in result and len(result['candidates']) > 0:
#             response_text = result['candidates'][0]['content']['parts'][0]['text']
            
#             # Check if the response is too short, and if so, ask for a longer response
#             lines = response_text.split('\n')
#             if len(lines) < 5:
#                 response_text += "\nCan you provide a more detailed answer?"
                
#             return response_text
#         else:
#             return "I couldn't generate a proper response. Please try again."
#     else:
#         print(f"Error generating response with Gemini API: {response.status_code}, {response.text}")
#         return "Error generating response."


# def main():
#     """Main workflow function."""
#     print("Starting complete workflow...")
    
#     # Initialize memory (store last 5 messages)
#     memory = deque(maxlen=5)
    
#     # Step 1: Extract text from PDF using Gemini
#     pdf_path = "data/raw_data/t2.pdf"
#     if not os.path.exists(pdf_path):
#         print(f"PDF file not found at {pdf_path}")
#         return
    
#     print("Step 1: Extracting text from PDF using Gemini API...")
#     extracted_text = extract_text_from_pdf_via_gemini(pdf_path)
    
#     if not extracted_text:
#         print("No text extracted from PDF.")
#         return
    
#     # Clean, split text, generate embeddings, and the rest
#     cleaned_text = clean_text(extracted_text)
#     chunks = split_text_into_chunks(cleaned_text)
#     embeddings = generate_embeddings_for_chunks(chunks)
    
#     embeddings_dir = "data/embeddings"
#     save_embeddings(embeddings, embeddings_dir)
    
#     faiss_index_path = 'data/embeddings/faiss_index.index'
#     index = create_faiss_index(embeddings)
#     save_faiss_index(index, faiss_index_path)
    
#     print("\nStep 7: Starting conversational interface...")
#     print("You can now ask questions about the document. Type 'quit' or 'exit' to end the conversation.")
    
#     while True:
#         query = input("\nAsk a question: ").strip()
        
#         if query.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break
        
#         if not query:
#             print("Please enter a question.")
#             continue
        
#         print("Searching for relevant information...")
        
#         indices = query_faiss_index(query, index, top_k=3)
#         if indices is not None:
#             print(f"Found {len(indices[0])} relevant chunks.")
#             response = generate_response_with_context(query, indices, embeddings_dir, memory)
#             print("\nAnswer:", response)
#         else:
#             print("No relevant information found.")

# if __name__ == "__main__":
#     main()
