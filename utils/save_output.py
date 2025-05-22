def save_result(query, gpt_response, filepath="output/result.json"):
    import json, os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "response": gpt_response
        }, f, ensure_ascii=False, indent=2)
