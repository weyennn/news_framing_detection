def build_contrastive_prompt(query, clusters, top_terms, max_examples=3):
    """
    Buat prompt untuk LLM berdasarkan hasil clustering dan topik utama.
    :param query: Pertanyaan dari user
    :param clusters: Dict[int, List[str]] - cluster id ke list dokumen
    :param top_terms: Dict[int, List[str]] - cluster id ke list topik dominan
    :param max_examples: Maks jumlah contoh dokumen per cluster
    :return: string prompt
    """
    prompt = f"Pertanyaan pengguna:\n{query}\n\nSistem mendeteksi dua kelompok narasi utama dari berita:\n"

    for cluster_id, docs in clusters.items():
        terms = ", ".join(top_terms.get(cluster_id, [])[:5])
        prompt += f"\n---\n📚 Cluster {cluster_id + 1} – Topik: {terms}\n\nContoh isi berita:\n"
        for i, doc in enumerate(docs[:max_examples]):
            excerpt = doc.strip().replace("\n", " ")[:300]
            prompt += f"{i+1}. \"{excerpt}...\"\n"

    prompt += ("""
---

Tolong:
1. Rangkum masing-masing narasi secara netral.
2. Jelaskan perbedaan sudut pandang yang muncul dari dua kelompok tersebut.
""")

    return prompt