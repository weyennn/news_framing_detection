import json
from retriever.bm25_retriever import bm25_retrieve
from clustering.contrastive_kmeans import cluster_documents_kmeans, get_top_terms_per_cluster
from llm.call_gpt import get_response_from_gpt
from llm.prompt_contrastive import build_contrastive_prompt
from utils.config_loader import load_config

# === LOAD CONFIG ===
config = load_config()
TOP_K = config["retriever"]["top_k"]
MODEL_NAME = config["llm"]["model"]
TEMP = config["llm"]["temperature"]
API_KEY = config["openrouter"]["api_key"]

# === LOAD DATA ===
with open("data/berita.json", encoding="utf-8") as f:
    data = json.load(f)

documents = [item["content"] for item in data if "content" in item and item["content"].strip() != ""]

# === LOOP INPUT QUERY ===
while True:
    query = input("\nMasukkan pertanyaan: ")
    retrieved_docs = bm25_retrieve(query, documents, top_k=TOP_K)
    if not retrieved_docs:
        print("❌ Tidak ada dokumen yang cocok. Coba query lain.")
        continue

    print(f"✅ Ditemukan {len(retrieved_docs)} dokumen dari BM25.")

    # === CLUSTERING ===
    labels, vectorizer, model = cluster_documents_kmeans(retrieved_docs, n_clusters=2)
    top_terms = get_top_terms_per_cluster(model, vectorizer, top_n=7)

    # === GROUP DOCS PER CLUSTER ===
    clusters = {i: [] for i in set(labels)}
    for idx, label in enumerate(labels):
        clusters[label].append(retrieved_docs[idx])

    # === BUILD PROMPT & CALL GPT ===
    prompt = build_contrastive_prompt(query, clusters, top_terms)
    print("\n📤 Prompt dikirim ke GPT...")
    response = get_response_from_gpt(prompt, api_key=API_KEY, model=MODEL_NAME)

    # === OUTPUT ===
    print("\n💬 Jawaban GPT-3.5:")
    print(response)

    break