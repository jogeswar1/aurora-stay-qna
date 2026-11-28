
import os
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_utils import retrieve, rag

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "faiss_index")

rag_api = Flask("Aurora Stay API")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

if not os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
    raise FileNotFoundError(
        f"FAISS index not found at {VECTORSTORE_PATH}. "
        "Ensure backend/faiss_index/index.faiss and index.pkl are present."
    )

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
print(f"Loaded FAISS vectors: {vectorstore.index.ntotal}")

def read_payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "Request body must be valid JSON."}), 400)

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return None, (jsonify({"error": "'query' is required and must be a non-empty string."}), 400)

    try:
        k = int(data.get("k", 8))
        max_tokens = int(data.get("max_tokens", 512))
        temperature = float(data.get("temperature", 0.0))
        top_p = float(data.get("top_p", 1.0))
    except (TypeError, ValueError):
        return None, (jsonify({"error": "k, max_tokens, temperature and top_p must be numeric."}), 400)

    if not 1 <= k <= 20:
        return None, (jsonify({"error": "k must be between 1 and 20."}), 400)
    if not 32 <= max_tokens <= 4096:
        return None, (jsonify({"error": "max_tokens must be between 32 and 4096."}), 400)
    if not 0.0 <= temperature <= 2.0 or not 0.0 < top_p <= 1.0:
        return None, (jsonify({"error": "temperature must be 0-2 and top_p must be >0-1."}), 400)

    return {
        "query": query.strip(),
        "k": k,
        "model_name": data.get("model_name", "gpt-4o-mini"),
        "temperature": temperature,
     #   "top_p": top_p,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }, None

@rag_api.get("/")
def home():
    return jsonify({"status": "ok", "service": "Aurora Stay RAG API"})

@rag_api.get("/health")
def health():
    return jsonify({"status": "healthy", "vectors": vectorstore.index.ntotal})

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    data, error = read_payload()
    if error:
        return error
    try:
        chunks = retrieve(data["query"], data["k"], vectorstore)
        return jsonify({
            "query": data["query"],
            "k": data["k"],
            "relevant_chunks": chunks,
        })
    except Exception as exc:
        rag_api.logger.exception("Retrieval failed")
        return jsonify({"error": "Retrieval failed", "details": str(exc)}), 500

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    data, error = read_payload()
    if error:
        return error
    try:
        answer, chunks = rag(
            query=data["query"],
            k=data["k"],
            model_name=data["model_name"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            max_tokens=data["max_tokens"],
            vectorstore=vectorstore,
        )
        return jsonify({
            **data,
            "relevant_chunks": chunks,
            "answer": answer,
        })
    except Exception as exc:
        rag_api.logger.exception("Answer generation failed")
        return jsonify({"error": "Answer generation failed", "details": str(exc)}), 500

@rag_api.post("/v1/answer")
def answer():
    data, error = read_payload()
    if error:
        return error
    try:
        generated_answer, _ = rag(
            query=data["query"],
            k=data["k"],
            model_name=data["model_name"],
            temperature=data["temperature"],
            top_p=data["top_p"],
            max_tokens=data["max_tokens"],
            vectorstore=vectorstore,
        )
        return jsonify({"answer": generated_answer})
    except Exception as exc:
        rag_api.logger.exception("Answer generation failed")
        return jsonify({"error": "Answer generation failed", "details": str(exc)}), 500

if __name__ == "__main__":
    rag_api.run(host="0.0.0.0", port=7860, debug=False)
