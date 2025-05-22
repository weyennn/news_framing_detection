def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    true_positives = [doc for doc in retrieved_k if doc in relevant]
    return len(true_positives) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    true_positives = [doc for doc in retrieved_k if doc in relevant]
    return len(true_positives) / len(relevant)

def average_precision(retrieved, relevant):
    hits = 0
    score = 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0

def mean_average_precision(all_retrieved, all_relevant):
    return sum(average_precision(r, all_relevant[q]) for q, r in all_retrieved.items()) / len(all_relevant)
