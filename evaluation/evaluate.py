import sys
import os
import json
import time
import torch
import numpy as np
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.qa_model import MultimodalQAAgent

# Paths
QUESTIONS_FILE = 'data/generated_chunk_questions.json'
RESULTS_FILE = 'evaluation/qa_results.json'
SUMMARY_FILE = 'evaluation/qa_summary.json'
CACHE_FILE = 'evaluation/gt_embeddings_cache.json'

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get embedding from OpenAI
def get_embedding(text):
    text = text.replace("\n", " ").strip()
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return torch.tensor(response.data[0].embedding)

# Load generated questions
with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

total = len(test_data)
print(f"Loaded {total} questions for evaluation.\n")

# Initialize QA agent
qa_agent = MultimodalQAAgent()

# Load or compute ground truth embeddings
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        gt_embeddings_list = json.load(f)
    gt_embeddings = [torch.tensor(e) for e in gt_embeddings_list]
    print(f"Loaded cached ground truth embeddings from {CACHE_FILE}")
else:
    gt_texts = [item.get('chunk_text', '') for item in test_data]
    gt_embeddings = [get_embedding(gt) for gt in gt_texts]
    # Save cache
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump([e.tolist() for e in gt_embeddings], f)
    print(f"Saved ground truth embeddings cache to {CACHE_FILE}")

# Evaluate each question
accurate_count = 0
response_times = []
similarities = []
results = []

for idx, item in enumerate(test_data, 1):
    question = item['question']
    gt_emb = gt_embeddings[idx - 1]

    start = time.time()
    try:
        result = qa_agent.ask(question)
        bot_answer = result.get('answer', '')
    except Exception as e:
        print(f"[{idx}/{total}] Error for question '{question}': {e}")
        continue
    end = time.time()
    response_time = end - start
    response_times.append(response_time)

    # Compute similarity
    bot_emb = get_embedding(bot_answer)
    similarity = torch.nn.functional.cosine_similarity(gt_emb, bot_emb, dim=0).item()
    similarities.append(similarity)

    # Consider accurate if similarity ≥ 0.7
    accurate = similarity >= 0.7
    if accurate:
        accurate_count += 1

    # Save detailed result
    results.append({
        "chunk_id": item.get("chunk_id"),
        "video_id": item.get("video_id"),
        "question": question,
        "ground_truth": item.get('chunk_text', ''),
        "bot_answer": bot_answer,
        "similarity": similarity,
        "accurate": accurate,
        "response_time": response_time
    })

    print(f"[{idx}/{total}] Question evaluated. Similarity: {similarity:.2f}")

# Summary metrics
avg_response_time = sum(response_times) / len(response_times) if response_times else 0
avg_similarity = sum(similarities) / len(similarities) if similarities else 0
accuracy = accurate_count / total if total else 0

summary = {
    "total_questions": total,
    "accuracy": accuracy,
    "avg_similarity": avg_similarity,
    "avg_response_time": avg_response_time
}

# Save results
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n===== Evaluation Complete =====")
print(f"Saved detailed results to {RESULTS_FILE}")
print(f"Saved summary metrics to {SUMMARY_FILE}")
print(f"Accuracy: {accuracy:.2%}, Avg Similarity: {avg_similarity:.2f}, Avg Response Time: {avg_response_time:.2f}s")
