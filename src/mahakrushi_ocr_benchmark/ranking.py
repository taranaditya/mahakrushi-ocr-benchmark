def rank_models(rows: list[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        entry = dict(row)
        if not entry.get("license_eligible", False) or not entry.get("runnable", True): entry.update(score=None, rank=None)
        else: entry["score"] = round((1-entry.get("cer", 1))*0.5 + entry.get("field_accuracy", 0)*0.3 + (1-entry.get("failure_rate", 1))*0.2, 4)
        ranked.append(entry)
    eligible = sorted((x for x in ranked if x["score"] is not None), key=lambda x: x["score"], reverse=True)
    for index, entry in enumerate(eligible, 1): entry["rank"] = index
    return sorted(ranked, key=lambda x: x["rank"] is None)
