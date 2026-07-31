import hashlib
import json


def stable_record_hash(record: dict, keys: list[str]) -> str:
    payload = {k: record.get(k) for k in keys}
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()