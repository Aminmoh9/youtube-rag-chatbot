import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_question(chunk_text, client):
    prompt = (
        "Read the following transcript chunk. If it is related to data analytics, SQL, databases, "
        "or technical concepts, generate a concise, relevant question that could be answered from this chunk. "
        "If it is not related, return an empty string.\n\n"
        f"Chunk:\n{chunk_text}\n\nQuestion:"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load chunks
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    results = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        chunk_text = chunk["text"]
        question = generate_question(chunk_text, client)

        if question:
            results.append({
                "chunk_id": chunk["chunk_id"],
                "video_id": chunk["video_id"],
                "question": question,
                "chunk_text": chunk_text
            })
            print(f"[{idx}/{total}] ✔ Question generated for {chunk['chunk_id']}")
            print(f"   → {question}\n")
        else:
            print(f"[{idx}/{total}] Skipped {chunk['chunk_id']} (not technical)")

    # Save output
    with open("data/generated_chunk_questions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTotal questions generated: {len(results)} out of {total} chunks.")


if __name__ == "__main__":
    main()
