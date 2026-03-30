# Automated QA System with LLM-Based Evaluation

A FastAPI-powered Question Answering system that generates answers and Q&A pairs from uploaded documents, evaluated using **Ragas** metrics (Faithfulness + Answer Relevancy) with Groq LLMs.

---

## 🚀 Features

- Upload any `.txt` document and ask questions about it
- Auto-generate Q&A pairs from a document
- LLM-based evaluation using **Ragas**:
  - **Faithfulness** — verifies answer is grounded in the document
  - **Answer Relevancy** — measures how relevant the answer is to the question
- Combined score with human-readable evaluation summary
- Interactive Swagger UI for testing

---

## 🏗️ System Flow

```
Swagger UI(User Interface)
    ↓
Upload Document (.txt)
    ↓
FastAPI receives request → decodes to context string
    ↓                              ↓
POST /ask                    POST /generate
User asks question           Specify num_pairs
    ↓                              ↓
Groq LLM                     Groq LLM
llama-3.1-8b-instant         llama-3.1-8b-instant
generates answer             generates Q&A pairs
    ↓                              ↓
Availability check           Parse JSON pairs
(info in document?)                ↓
    ↓                              ↓
    └──────────┬────────────────────┘
               ↓
       Ragas Evaluation
       qwen-2.5-32b (eval LLM)
       ┌─────────────┬──────────────────┐
       Faithfulness          Answer Relevancy
       LLM claim             LLM + MiniLM
       check                 embeddings
                       ↓
       Scores: faithfulness · relevancy · combined
                       ↓
       JSON response → Swagger UI
```

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| API Framework | FastAPI |
| Answer Generation | Groq — `llama-3.1-8b-instant` |
| Evaluation LLM | Groq — `qwen-2.5-32b` |
| Evaluation Framework | Ragas |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM Wrappers | LangChain (langchain-groq, langchain-huggingface) |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone (https://github.com/Kalyaani-hub/Automated-QA-generator-for-Text-Datasets)
cd Automated-QA-generator-for-Text-Datasets
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Copy the example env file and add your Groq API key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## ▶️ Running the App

```bash
uvicorn main:app --reload
```

Then open your browser at:
```
http://127.0.0.1:8000/docs
```

This opens the **Swagger UI** where you can test both endpoints.

---

## 📡 API Endpoints

### `POST /ask`
Upload a document and ask a question.

**Form inputs:**
| Field | Type | Description |
|---|---|---|
| `file` | `.txt` file | Document to search |
| `question` | string | Your question |

**Response:**
```json
{
  "question": "What is machine learning?",
  "answer": "Machine learning is...",
  "faithfulness_score": 0.92,
  "relevancy_score": 0.87,
  "combined_score": 0.89,
  "evaluation_summary": "Excellent (F: 0.92, R: 0.87) — LLM verified high quality",
  "evaluation_method": "LLM-based (Ragas with qwen-2.5-32b)"
}
```

---

### `POST /generate`
Auto-generate Q&A pairs from a document.

**Form inputs:**
| Field | Type | Description |
|---|---|---|
| `file` | `.txt` file | Document to generate from |
| `num_pairs` | int | Number of pairs to generate (default: 5) |

**Response:**
```json
{
  "total_pairs": 5,
  "average_faithfulness": 0.88,
  "average_relevancy": 0.84,
  "average_combined": 0.86,
  "evaluation_method": "LLM-based (Ragas with qwen-2.5-32b)",
  "pairs": [
    {
      "question": "...",
      "answer": "...",
      "faithfulness_score": 0.90,
      "relevancy_score": 0.85,
      "combined_score": 0.87
    }
  ]
}
```

---

## 📊 Evaluation Metrics

| Score Range | Label | Meaning |
|---|---|---|
| 0.8 – 1.0 | Excellent | LLM verified high quality |
| 0.6 – 0.8 | Good | LLM confirmed accuracy |
| 0.4 – 0.6 | Fair | LLM found partial support |
| 0.0 – 0.4 | Poor | LLM identified issues |

> **Note:** If the answer is not found in the document, scores are returned as `null` and evaluation is skipped.

---

## 📁 Project Structure

```
├── main.py                  # Main FastAPI application
├── requirements.txt         # Python dependencies
├── .env                     # Your API keys (never commit this)
├── .env.example             # Template for environment variables
├── sample_document.txt      # Sample document for testing
└── README.md
```

---

## ⚠️ Important Notes

- Only `.txt` files are supported currently
- Keep documents short (under ~2000 characters) for best evaluation accuracy — longer documents can cause the eval LLM to hit token limits
- The evaluation step can take 30–60 seconds depending on document length and Groq rate limits

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

[MIT](LICENSE)
