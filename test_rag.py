from rag import retrieve
from llm import generate_answer

question = """
What is the typical ground contact time during the acceleration phase of sprinting?
"""

top_docs, context_text = retrieve(question)

print("\n" + "=" * 100)
print("FULL CONTEXT")
print("=" * 100)
print(context_text)

answer = generate_answer(question, context_text)

print("\n" + "=" * 100)
print("LLM ANSWER")
print("=" * 100)
print(answer)