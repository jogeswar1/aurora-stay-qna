
import os
from langchain_openai import ChatOpenAI

INSUFFICIENT_ANSWER = (
    "The available policy documents do not contain enough information to answer this question."
)

def retrieve(query: str, k: int = 5, vectorstore=None) -> list[str]:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if vectorstore is None:
        raise ValueError("vectorstore is required")

    k = max(1, min(int(k), 20))
    results = vectorstore.similarity_search(query, k=k)

    return [
        (
            f"Source: {doc.metadata.get('source', 'unknown')}\n"
            f"Content type: {doc.metadata.get('content_type', 'unknown')}\n"
            f"Content:\n{doc.page_content}"
        )
        for doc in results
    ]

def generate(query: str, retrieved_chunks: list[str], model) -> str:
    context = "\n\n--- Retrieved passage ---\n\n".join(retrieved_chunks)
    prompt = f"""You are an assistant answering questions about AuroraStay hotel policies.

Question:
{query}

Retrieved policy passages:
{context}

Instructions:
- Answer only from the retrieved passages. Do not use outside knowledge or assumptions.
- Treat table labels and values as connected; interpret each value in the context of its column/header labels.
- Include relevant amounts, units, property names, conditions, exceptions, and limitations when present.
- If the passages do not support a complete answer, say what is missing.
- If the passages conflict, explain the conflict and identify the sources if possible.
- Do not claim a policy does not exist merely because one retrieved passage omits it.
- Be concise and direct.

If the retrieved passages do not contain enough information, say:
"{INSUFFICIENT_ANSWER}"
"""
    return model.invoke(prompt)

def rag(
    query: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None,
):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key or not api_base:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_API_BASE are not configured.")

    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=api_base,
    )

    chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=chunks, model=model)
    return answer.content, chunks
