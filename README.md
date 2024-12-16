# Llama2-7B Q&A System

## Overview
The Llama2-7B Q&A System is an interactive Streamlit-based application that leverages the Llama2-7B model to provide answers to user questions. It supports file uploads, URL-based content, and flexible answer formatting.

### Features
- **Multi-format Document Parsing**: Supports `.txt`, `.pdf`, `.docx`, `.pptx`, and `.xlsx` files.
- **Web Content Extraction**: Fetch and process textual content from web links.
- **Chunk-Based Context Handling**: Splits documents into manageable chunks for efficient processing.
- **Customizable Answer Formats**: Choose between default, bullet points, summaries, or answers with specific word limits.
- **Interactive Chat Interface**: Displays a history of questions and answers.

## Installation

### Prerequisites
- Python 3.8+
- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co)
- [Replicate](https://replicate.com)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/llama2-qa-system.git
   cd llama2-qa-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables by creating a `.env` file:
   ```plaintext
   REPLICATE_API_TOKEN=Use your token
   HUGGINGFACE_API_TOKEN=Use your token
   ```

4. Run the application:
   ```bash
   streamlit run llama2_qa_app.py
   ```

5. Open the application in your browser at `http://localhost:8501`.

## Usage

### File Upload
1. Drag and drop files into the file uploader in the sidebar.
2. Ensure the total file size is under 15 MB.

### Web Links
1. Paste comma-separated URLs into the text box in the sidebar.
2. The application will fetch and process the content.

### Question and Answer
1. Type your question into the text input box.
2. Select your preferred answer format (default, bullet points, summary, or specific length).
3. Click "Get Answer" to view the response.

### Chat History
The application maintains a session-specific chat history for reference.

## Development

### Code Structure
- `llama2_qa_app.py`: Main application file.
- `requirements.txt`: Python dependencies.

### Key Libraries
- `Streamlit`: Interactive user interface.
- `HuggingFaceEmbeddings`: Embedding model for text similarity.
- `FAISS`: Vector store for efficient document retrieval.
- `PyMuPDF`, `python-docx`, `python-pptx`, `pandas`: File parsers.

## Deployment
Deploy the app using services like [Streamlit Cloud](https://streamlit.io/cloud).

## License
See the LICENSE file for details.

## Contact
For issues or contributions, please create an issue in the [GitHub repository](https://github.com/your-repo/llama2-qa-system).
