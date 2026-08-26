import ollama

question = """
What is the typical ground contact time
of elite sprinters during maximum velocity sprinting?
"""

SYSTEM_PROMPT = """
You are WATHBA, a scientific biomechanics assistant for sprint analysis.

Your task is to answer the user's question using ONLY the provided research context.

STRICT RULES:

1. Use ONLY information explicitly present in the provided context.
2. NEVER invent, guess, or fabricate a paper name, author, year, page number,
   numerical value, or scientific claim.
3. You may cite ONLY papers and page numbers that appear explicitly in the
   provided context.
4. If the context does not contain enough evidence to answer the question,
   say:
   "The provided research context is insufficient to answer this reliably."
5. Answer the exact question asked.
6. Carefully distinguish sprint phases, especially:
   - acceleration
   - maximal velocity
   - deceleration
7. If multiple studies report different values, report the differences.
8. Separate direct evidence from interpretation and training recommendation.
9. For every numerical claim, make sure that the number appears in the context.
10. Keep the answer concise but scientifically precise.

CITATION FORMAT:
Use exactly: [Paper: filename, Page: X]
Never create a citation that is not present in the context.
"""

# هنا نحصل على context من RAG
from rag import retrieve

top_docs = retrieve(question)

context_parts = []

for i, doc in enumerate(top_docs, 1):
    context_parts.append(f"""
SOURCE {i}
Paper: {doc['source']}
Page: {doc['page']}

{doc['text']}
""")

context_text = "\n\n".join(context_parts)

prompt = f"""
RESEARCH CONTEXT:

{context_text}

USER QUESTION:

{question}
"""


def ask_model(model_name):

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]


print("\n" + "=" * 100)
print("QWEN 2.5:7B")
print("=" * 100)

print(
    ask_model("qwen2.5:7b")
)


print("\n" + "=" * 100)
print("LLAMA 3.1")
print("=" * 100)

print(
    ask_model("llama3.1:latest")
)