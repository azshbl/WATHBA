import requests
import json


OLLAMA_URL = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = """
You are WATHBA, a scientific biomechanics assistant for sprint analysis.

You may receive up to two sources of information:

1. RESEARCH CONTEXT: excerpts from peer-reviewed sprint biomechanics papers.
2. ATHLETE DATA: biomechanical KPIs extracted by computer vision from a
   specific athlete's own race video (e.g. stride length, ground contact
   time, knee angle, trunk lean). These numbers are measured, not invented
   by you — treat them as ground truth about THIS athlete.

STRICT RULES:

1. Use ONLY information explicitly present in the Research Context or the
   Athlete Data provided. NEVER invent, guess, or fabricate a paper name,
   author, year, page number, numerical value, or scientific claim.

2. You may cite ONLY papers and page numbers that appear explicitly in the
   Research Context. Athlete Data does not need a citation — attribute it
   to "the athlete's video analysis" instead.

3. If the Research Context does not contain enough evidence to support a
   comparison, say exactly:
   "The provided research context is insufficient to answer this reliably."

4. When comparing the athlete to research benchmarks:
   - State the athlete's own value first (from Athlete Data).
   - State the relevant research benchmark next (with citation).
   - Then describe how the athlete compares (faster/slower, longer/shorter,
     within typical elite range, etc.) — only using comparisons the numbers
     actually support.

5. Carefully distinguish sprint phases: acceleration, maximal velocity,
   deceleration. Never compare an athlete's acceleration-phase value to a
   maximal-velocity research benchmark, or vice versa.

6. Distinguish between population-level values, individual athlete values,
   extreme/best-case values, and correlations in the Research Context.
   When the question asks for "typical" or "average", use population-level
   values as the main benchmark — not an individual extreme value.

7. Keep the answer concise and scientifically precise.

CITATION FORMAT (research only):
[Paper: filename, Page: X]

Never create a citation that is not present in the Research Context.
"""


def format_athlete_data(athlete_data: dict) -> str:
    """
    Turn the computer-vision JSON output into a clean, readable block
    the LLM can use directly in the prompt.

    Expected shape (adapt keys to match your CV pipeline's actual output):
    {
        "athlete_name": "Ahmed A.",
        "clip_id": "race_2026_08_20_01",
        "phase": "maximum_velocity",       # acceleration | maximum_velocity | deceleration
        "metrics": {
            "ground_contact_time_s": 0.098,
            "stride_length_m": 2.21,
            "stride_frequency_hz": 4.55,
            "knee_angle_deg": 148,
            "trunk_lean_deg": 6.2,
            "speed_ms": 10.4
        }
    }
    """
    if not athlete_data:
        return "No athlete data provided for this question."

    lines = [
        f"Athlete: {athlete_data.get('athlete_name', 'Unknown')}",
        f"Clip ID: {athlete_data.get('clip_id', 'N/A')}",
        f"Sprint Phase: {athlete_data.get('phase', 'N/A')}",
        "Measured KPIs:",
    ]

    for key, value in athlete_data.get("metrics", {}).items():
        readable_key = key.replace("_", " ")
        lines.append(f"  - {readable_key}: {value}")

    return "\n".join(lines)


def generate_answer(question, context_text, model, athlete_data=None):
    """
    athlete_data: optional dict from the computer vision pipeline.
    Pass None (default) for pure research questions with no athlete clip.
    """

    athlete_block = format_athlete_data(athlete_data) if athlete_data else None

    sections = [f"Research Context:\n-----------------\n{context_text}\n-----------------"]

    if athlete_block:
        sections.append(f"Athlete Data:\n-------------\n{athlete_block}\n-------------")

    sections.append(f"Question:\n{question}")
    sections.append("Answer the question based strictly on the information above.")

    user_prompt = "\n\n".join(sections)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=300,
    )

    response.raise_for_status()
    data = response.json()

    return data["message"]["content"]


# ------------------------------------------------------------------
# Example usage (what test_rag.py would look like tomorrow)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from rag import retrieve

    question = "How does this athlete's ground contact time compare to elite sprinters at maximum velocity?"

    athlete_data = {
        "athlete_name": "Test Athlete",
        "clip_id": "demo_clip_01",
        "phase": "maximum_velocity",
        "metrics": {
            "ground_contact_time_s": 0.098,
            "stride_length_m": 2.21,
            "speed_ms": 10.4,
        },
    }

    top_docs, context_text = retrieve(question)

    answer = generate_answer(
        question=question,
        context_text=context_text,
        model="llama3.1:latest",
        athlete_data=athlete_data,
    )

    print(answer)
