"""
QA System with LLM-Based Evaluation 
Uses Ragas Faithfulness + Answer Relevancy with proper configuration

Run:  uvicorn main_final_working:app --reload

Install:
    pip install fastapi uvicorn groq ragas langchain-groq datasets
    pip install langchain-huggingface sentence-transformers
"""

import asyncio
import json
import math
import os

from groq import AsyncGroq
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel, Field

from ragas import evaluate, EvaluationDataset
try:
    from ragas.dataset_schema import SingleTurnSample
except ImportError:
    from ragas import SingleTurnSample

from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ── Setup ──────────────────────────────────────────────────────────────────────
print(" Initializing LLM-based evaluation system...")

GROQ_API_KEY = os.environ["GROQ_API_KEY"] # THIS MODEL IS USED FOR PROJECT PURPOSE ONLY . YOU CAN USE YOUR OWN MODEL AND API KEY.
client = AsyncGroq(api_key=GROQ_API_KEY)

ANSWER_MODEL = "llama-3.1-8b-instant"  # THIS MODEL IS USED FOR PROJECT PURPOSE ONLY . YOU CAN USE YOUR OWN MODEL.
EVAL_MODEL = "qwen/qwen3-32b"    # THIS MODEL IS USED FOR PROJECT PURPOSE ONLY . YOU CAN USE YOUR OWN MODEL.

print(f" Answer Model: {ANSWER_MODEL}")
print(f" Evaluation Model: {EVAL_MODEL}")

# LLM for Ragas
ragas_llm = LangchainLLMWrapper(
    ChatGroq(
        model=EVAL_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        timeout=120,
        max_tokens=4096
    )
)

# Embeddings for Answer Relevancy
print(" Loading embeddings model...")
try:
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
    print(" Embeddings loaded successfully!")
    
    # Test embeddings
    test_vec = embeddings_model.embed_query("test")
    print(f" Embeddings test passed! Dimension: {len(test_vec)}")
except Exception as e:
    print(f" Error loading embeddings: {e}")
    raise

print(" All models loaded!\n")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QA System - LLM-Based Evaluation",
    description="""
## Automated QA Generator with LLM-Based Evaluation

### Metrics:
- **Faithfulness (Ragas)**: LLM verifies answer grounding
- **Answer Relevancy (Ragas)**: LLM + embeddings measure relevance

### Endpoints:
- POST /ask: Upload file + ask question
- POST /generate: Generate Q&A pairs from document

Fully LLM-based evaluation satisfies project requirements!
""",
)

# ── Schemas ────────────────────────────────────────────────────────────────────

class AskResponse(BaseModel):
    question: str
    answer: str
    faithfulness_score: float | None = None
    relevancy_score: float | None = None
    combined_score: float | None = None
    evaluation_summary: str
    evaluation_method: str = Field(default="LLM-based (Ragas)")

class QAPair(BaseModel):
    question: str
    answer: str
    faithfulness_score: float
    relevancy_score: float
    combined_score: float

class GenerateResponse(BaseModel):
    total_pairs: int
    average_faithfulness: float
    average_relevancy: float
    average_combined: float
    evaluation_method: str = Field(default="LLM-based (Ragas)")
    pairs: list[QAPair]

# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(value, default=0.0):
    """Convert value to float, handling NaN and None cases."""
    try:
        if value is None:
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _parse_json(raw: str):
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return json.loads(clean.strip())


def _evaluation_summary(faithfulness: float | None, relevancy: float | None, answer: str) -> str:
    """Generate evaluation summary."""
    if faithfulness is None or relevancy is None:
        if "not available" in answer.lower() or "not in the" in answer.lower():
            return "N/A — Information not found in document"
        return "Could not evaluate"
    
    combined = (faithfulness + relevancy) / 2
    
    if combined >= 0.8:
        return f"Excellent (F: {faithfulness:.2f}, R: {relevancy:.2f}) — LLM verified high quality"
    elif combined >= 0.6:
        return f"Good (F: {faithfulness:.2f}, R: {relevancy:.2f}) — LLM confirmed accuracy"
    elif combined >= 0.4:
        return f"Fair (F: {faithfulness:.2f}, R: {relevancy:.2f}) — LLM found partial support"
    else:
        return f"Poor (F: {faithfulness:.2f}, R: {relevancy:.2f}) — LLM identified issues"


async def _get_answer(context: str, question: str) -> str:
    """Generate answer using LLM."""
    response = await client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a comprehensive Q&A assistant. "
                    "Answer ONLY from the provided context. "
                    "If the answer is not in the context say: "
                    "'This information is not available in the uploaded document.' "
                    "\n\n"
                    "Provide detailed, well-written answers in natural prose:\n"
                    "- Start with a clear definition or main explanation\n"
                    "- Add supporting details, examples, or context from the document\n"
                    "- Include relevant additional information if available\n"
                    "- Write in flowing paragraphs (4-6 sentences total)\n"
                    "\n"
                    "Write in natural language without section labels."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=2048,
        timeout=60,
    )
    return response.choices[0].message.content.strip()


async def _generate_pairs(context: str, num_pairs: int) -> list[dict]:
    """Generate Q&A pairs using LLM."""
    response = await client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Q&A pair generator. "
                    "Generate high-quality question-answer pairs that:\n"
                    "1. Questions are clear and specific\n"
                    "2. Answers use EXACT phrases from the context\n"
                    "3. Each answer is 1-2 sentences, directly from the text\n"
                    "4. Focus on factual, verifiable information\n"
                    "\n"
                    "Return ONLY valid JSON array, no markdown, no explanations."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Generate exactly {num_pairs} question-answer pairs.\n"
                    "Make answers SHORT and GROUNDED in the text.\n"
                    f"Return ONLY this JSON array:\n"
                    '[{"question":"...","answer":"..."}]'
                )
            }
        ],
        max_tokens=4096,
        temperature=0.2,  # Lower for more grounded answers
        timeout=120,
    )
    return _parse_json(response.choices[0].message.content)


def _ragas_score_with_llm(context: str, pairs: list[dict]) -> dict:
    """
    Run Ragas evaluation with BOTH metrics in a single call.
    """
    samples = [
        SingleTurnSample(
            user_input=p["question"],
            response=p["answer"],
            retrieved_contexts=[context],
        )
        for p in pairs
    ]
    
    try:
        print(f"🔍 Running Ragas evaluation on {len(pairs)} pairs...")
        
        # Run BOTH metrics together in ONE evaluation call
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=[
                Faithfulness(llm=ragas_llm),
                AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
            ],
        )
        
        df = result.to_pandas()
        print(f"✅ Evaluation complete!")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Extract scores
        faithfulness_scores = []
        relevancy_scores = []
        
        for idx, row in df.iterrows():
            f_val = row.get("faithfulness", 0.0)
            r_val = row.get("answer_relevancy", 0.0)
            
            f_score = round(_safe_float(f_val, default=0.0), 2)
            r_score = round(_safe_float(r_val, default=0.0), 2)
            
            print(f"Sample {idx}: Faithfulness={f_score}, Relevancy={r_score}")
            
            faithfulness_scores.append(f_score)
            relevancy_scores.append(r_score)
        
        return {
            "faithfulness": faithfulness_scores,
            "relevancy": relevancy_scores
        }
        
    except Exception as e:
        print(f"❌ Error in Ragas evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "faithfulness": [0.0] * len(pairs),
            "relevancy": [0.0] * len(pairs)
        }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    raw_bytes = await file.read()
    try:
        context = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        context = raw_bytes.decode("latin-1")

    if not context.strip():
        raise HTTPException(status_code=400, detail="Empty file")

    # Generate answer
    try:
        answer = await _get_answer(context, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    # Check if not available
    is_not_available = (
        "not available" in answer.lower() or 
        "not in the" in answer.lower() or
        "not found in" in answer.lower()
    )
    
    if is_not_available:
        faithfulness_score = None
        relevancy_score = None
        combined_score = None
    else:
        # LLM-based evaluation
        try:
            scores = await asyncio.to_thread(
                _ragas_score_with_llm,
                context,
                [{"question": question, "answer": answer}]
            )
            faithfulness_score = scores["faithfulness"][0]
            relevancy_score = scores["relevancy"][0]
            combined_score = round((faithfulness_score + relevancy_score) / 2, 2)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            faithfulness_score = 0.0
            relevancy_score = 0.0
            combined_score = 0.0

    return AskResponse(
        question=question,
        answer=answer,
        faithfulness_score=faithfulness_score,
        relevancy_score=relevancy_score,
        combined_score=combined_score,
        evaluation_summary=_evaluation_summary(faithfulness_score, relevancy_score, answer),
        evaluation_method=f"LLM-based (Ragas with {EVAL_MODEL})"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    file: UploadFile = File(...),
    num_pairs: int = Form(default=5), #you can keep any number of QA pairs. 
):
    raw_bytes = await file.read()
    try:
        context = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        context = raw_bytes.decode("latin-1")

    if not context.strip():
        raise HTTPException(status_code=400, detail="Empty file")

    # Generate pairs
    try:
        raw_pairs = await _generate_pairs(context, num_pairs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    if not raw_pairs:
        raise HTTPException(status_code=500, detail="No pairs generated")

    # LLM-based evaluation
    try:
        scores = await asyncio.to_thread(_ragas_score_with_llm, context, raw_pairs)
        faithfulness_scores = scores["faithfulness"]
        relevancy_scores = scores["relevancy"]
    except Exception as e:
        print(f"Evaluation error: {e}")
        faithfulness_scores = [0.0] * len(raw_pairs)
        relevancy_scores = [0.0] * len(raw_pairs)

    pairs = [
        QAPair(
            question=r["question"],
            answer=r["answer"],
            faithfulness_score=f_score,
            relevancy_score=r_score,
            combined_score=round((f_score + r_score) / 2, 2)
        )
        for r, f_score, r_score in zip(raw_pairs, faithfulness_scores, relevancy_scores)
    ]

    avg_f = round(sum(p.faithfulness_score for p in pairs) / len(pairs), 2) if pairs else 0.0
    avg_r = round(sum(p.relevancy_score for p in pairs) / len(pairs), 2) if pairs else 0.0
    avg_c = round(sum(p.combined_score for p in pairs) / len(pairs), 2) if pairs else 0.0

    return GenerateResponse(
        total_pairs=len(pairs),
        average_faithfulness=avg_f,
        average_relevancy=avg_r,
        average_combined=avg_c,
        evaluation_method=f"LLM-based (Ragas with {EVAL_MODEL})",
        pairs=pairs,
    )