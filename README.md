
# Groq-HF-Conversational-Bot

This project implements a Conversational Retrieval-Augmented Generation (RAG) application using HuggingFace embeddings and the Groq Gemma2-9b-It model. Users can upload PDF documents, transform their content into vector embeddings, and interact with them conversationally. Leveraging chat history for enhanced context-aware responses, it delivers a seamless, intelligent question-answering experience.

# Features

### 1. PDF Upload and Processing
- Upload one or more PDF documents through a drag-and-drop interface or file selector.
- Process uploaded PDFs using `PyPDFLoader` to extract text accurately.

### 2. Chunk-Based Processing
- Use `RecursiveCharacterTextSplitter` to divide large documents into smaller, manageable chunks.
- Each chunk can contain up to **5000 characters** with an overlap of **200 characters** to maintain context between chunks.

### 3. Embedding Creation
- Generate high-quality embeddings using the **HuggingFace `all-MiniLM-L6-v2` model**.
- Embeddings are optimized for semantic search and similarity-based retrieval.

### 4. Vector Storage
- Store embeddings efficiently in **Chroma**, a high-performance vector database.
- Retrieve the most relevant document chunks quickly using similarity search.

### 5. Chat History Management
- Maintain session-specific chat histories using `ChatMessageHistory`.
- Reformulate user queries based on previous interactions for enhanced contextual understanding.

### 6. Language Model Integration
- Use the **Groq LLM (Gemma2-9b-It)** for generating concise, contextually relevant responses to user queries.

### 7. Interactive Chat Interface
- Developed with **Streamlit**, offering a user-friendly, responsive interface.
- Input queries, review responses, and track ongoing chat history.

### 8. Standalone Question Refinement
- Automatically rephrase user queries into standalone questions when necessary, ensuring accurate retrieval from document embeddings.

## Run Locally
#### Prerequisites
Ensure you have the following software installed:

- Python 3.12: Ensure Python is installed and updated to version 3.12 or later.
- pip (Python package manager)
- API Keys:
    - HuggingFace API Key (for embedding generation).
    - Groq API Key (for LLM responses).


### Setup Instructions

#### 1. Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/pruthi-saksham/Groq-HF-Conversational-Bot.git
```

Go to the project directory

```bash
cd Groq-HF-Conversational-Bot
```

#### 2. Install Dependencies
Install the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### 3. Set Up Environment Variables
Create a `.env` file in the root directory and configure the HuggingFace API key:

```bash
HF_API_KEY=your_huggingface_api_key
```
*The Groq API Key will be entered directly in the application when prompted.*



#### 4. Start the Application
Run the application using Streamlit:

```bash
streamlit run app.py
```


# Project Architecture

#### Frontend
- **Streamlit**: Provides the user interface for PDF upload and interaction.

#### Backend
- **Text Extraction**: `PyPDFLoader` extracts text from uploaded PDFs.
- **Text Splitting**: `RecursiveCharacterTextSplitter` divides text into smaller chunks.
- **Embeddings**: `HuggingFaceEmbeddings` generates vector embeddings.
- **Vector Storage**: `Chroma` stores embeddings and retrieves relevant chunks.
- **LLM Integration**: `Groq` LLM generates responses based on user queries and retrieved document context.

---



# Tech Stack

This project leverages the following tools and technologies:

#### Language
- **Python**: The core programming language used for both backend logic and frontend integration.

---

#### Frameworks
- **Streamlit**: 
  - **Purpose**: Provides an interactive and user-friendly web interface for uploading files and querying the system.
  - **Features**: Real-time updates, session management, and component-rich design.

---

#### Libraries
- **LangChain**:
  - **Purpose**: Facilitates retrieval-augmented generation (RAG) by connecting vector databases, language models, and user prompts.
  - **Functions Used**:
    - `create_history_aware_retriever`: Adds context to queries based on chat history.
    - `create_retrieval_chain`: Orchestrates the retrieval and question-answering workflow.
    - `create_stuff_documents_chain`: Merges retrieved documents into a cohesive context.

- **Groq**:
  - **Purpose**: A high-performance language model for generating intelligent and concise responses to user queries.
  - **Integration**: Utilized through the `ChatGroq` class.

- **HuggingFace Transformers**:
  - **Purpose**: Provides pre-trained models for generating semantic embeddings from document text.
  - **Model Used**: `all-MiniLM-L6-v2`.

- **Chroma**:
  - **Purpose**: A high-performance vector database for storing and retrieving embeddings.
  - **Functions**:
    - `from_documents`: Creates a vector store from processed document embeddings.
    - `as_retriever`: Fetches the most relevant document chunks based on queries.

- **PyPDFLoader**:
  - **Purpose**: Extracts text content from PDF files.
  - **Features**: Handles complex PDF layouts for reliable text extraction.

- **RecursiveCharacterTextSplitter**:
  - **Purpose**: Splits large documents into smaller, overlapping chunks for efficient processing.
  - **Settings**: Chunk size of 5000 characters with a 200-character overlap.

- **dotenv**:
  - **Purpose**: Loads environment variables securely from a `.env` file.
  - **Functionality**: Manages sensitive data like API keys.

---

#### Functions and Utilities

This project relies on various functions and utilities from the included libraries and frameworks:

---

**Chat History**
- `ChatMessageHistory`:
  - **Purpose**: Manages session-specific chat histories, storing user inputs and assistant responses.
  - **Usage**: Ensures contextual awareness across a single session.

- **`RunnableWithMessageHistory`**:
  - **Purpose**: Links the RAG chain with chat histories, enabling stateful interaction.
  - **Usage**: Associates queries with historical context for personalized responses.

---

**Prompt Templates**
- `ChatPromptTemplate`:
  - **Purpose**: Structures and organizes prompts for the LLM, including:
    - Reformulating user queries to standalone questions.
    - Generating concise and accurate answers.
  - **Usage**: Creates customizable, template-driven prompts for both retrieval and answering.

- `MessagesPlaceholder`:
  - **Purpose**: Marks where the chat history should be dynamically inserted in prompts.
  - **Usage**: Enables dynamic addition of chat context to prompt templates.

---

**Environment Management**
- `load_dotenv`:
  - **Purpose**: Ensures secure loading of sensitive data like API keys from a `.env` file.
  - **Usage**: Used during initialization to set up HuggingFace and Groq API keys.

- `os.getenv`:
  - **Purpose**: Fetches environment variable values from the system or `.env` file.
  - **Usage**: Retrieves the API keys for HuggingFace embeddings and Groq.

---

**File Handling**
- `st.file_uploader`:
  - **Purpose**: Allows users to upload PDF files directly from the Streamlit interface.
  - **Usage**: Handles multiple file uploads for processing.

- *Temporary File Storage*:
  - **Purpose**: Temporarily saves uploaded PDF files for runtime operations.
  - **Usage**: Ensures uploaded files are accessible for text extraction.

---

**PDF Processing**
- `PyPDFLoader.load`:
  - **Purpose**: Extracts text content from uploaded PDF files.
  - **Usage**: Processes all uploaded files and converts them into LangChain-compatible documents.

---

**Document Splitting**
- `RecursiveCharacterTextSplitter.split_documents`:
  - **Purpose**: Splits long documents into smaller, overlapping chunks for embedding generation.
  - **Settings**: Chunk size = 5000 characters; Overlap = 200 characters.
  - **Usage**: Ensures efficient and context-aware processing of large files.

---

**Vector Store and Retrieval**
- `Chroma.from_documents`:
  - **Purpose**: Creates a vector database from document embeddings.
  - **Usage**: Stores the generated embeddings for semantic similarity searches.

- `Chroma.as_retriever`:
  - **Purpose**: Converts the vector database into a retriever for querying.
  - **Usage**: Fetches the most relevant document chunks for user queries.

---

**Retrieval-Augmented Generation (RAG)**
- `create_history_aware_retriever`:
  - **Purpose**: Combines a retriever with history-aware query reformulation.
  - **Usage**: Improves retrieval accuracy by using contextual user queries.

- `create_retrieval_chain`:
  - **Purpose**: Combines the retriever with a language model for end-to-end query answering.
  - **Usage**: Orchestrates document retrieval and response generation.

- `create_stuff_documents_chain`:
  - **Purpose**: Processes retrieved document chunks into a unified context for LLM prompts.
  - **Usage**: Enables precise and context-rich answers from the LLM.

---

**Streamlit Integration**
- `st.title`:
  - **Purpose**: Displays the application title.
  - **Usage**: Provides a header for the Streamlit app.

- `st.write`:
  - **Purpose**: Adds explanatory text and information to the interface.
  - **Usage**: Guides users on how to interact with the app.

- `st.text_input`:
  - **Purpose**: Collects user inputs such as API keys, session IDs, and questions.
  - **Usage**: Creates interactive fields for user input.

- `st.success`:
  - **Purpose**: Displays the assistant's responses in a success message format.
  - **Usage**: Enhances readability of responses.

- `st.warning`:
  - **Purpose**: Alerts users about missing API keys or other important issues.
  - **Usage**: Improves user experience by providing actionable feedback.

---

This comprehensive set of functions ensures smooth operation, robust document processing, and an interactive user experience.


# 🚀 About Me
*Hi, I’m Saksham Pruthi, an AI Engineer passionate about creating innovative AI-powered solutions. I specialize in Generative AI, designing systems that bridge cutting-edge research and practical applications. With expertise in various AI frameworks and an eye for scalable technology, I enjoy tackling challenging projects that drive real-world impact.*
## 🛠 Skills
+ **Programming Languages**: Python, C++
+ **Generative AI Technologies**:  Proficient in deploying and fine-tuning a variety of LLMs including Llama2, GPT (OpenAI), Mistral, Gemini Pro  using frameworks like Hugging Face, OpenAI,Groq and Groq. Expertise in NLP tasks like tokenization, sentiment analysis, summarization, and machine translation. Skilled in computer vision (CV) with models for image classification, object detection, and segmentation (YOLO). Expertise in MLOps, building and maintaining pipelines for model training and monitoring. Proficient in conversational AI with platforms LangChain. Skilled in synthetic data generation and code generation
+ **Vector Databases and Embedding Libraries**: Proficient in ChromaDB and FAISS for efficient vector storage, retrieval, and similarity search.
+ **Frameworks, Tools & Libraries**: LangChain, HuggingFace , OpenAI API, Groq, TensorFlow, PyTorch, Streamlit.
+ **Databases**: MongoDB, ChromaDB
+ **Version Control**: Proficient in using Git for version control and GitHub for collaborative development, repository management, and continuous integration workflows.
