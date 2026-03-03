"""Convert JSONL conversation files to SFT JSON format.

Reads orig_jsonls/conversations_n200_s{0,1}_shorter.jsonl and produces:
  - jsons/s0.json        (s0 only)
  - jsons/s1.json        (s1 only)
  - jsons/s0_s1_merge.json  (both merged)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

BASE_DIR = Path(__file__).resolve().parent
ORIG_DIR = BASE_DIR / "orig_jsonls"
OUT_DIR = BASE_DIR / "jsons"

SOURCES = [
    {
        "path": ORIG_DIR / "conversations_n200_s0_shorter.jsonl",
        "output": OUT_DIR / "s0.json",
    },
    {
        "path": ORIG_DIR / "conversations_n200_s1_shorter.jsonl",
        "output": OUT_DIR / "s1.json",
    },
]

MERGED_OUTPUT = OUT_DIR / "s0_s1_merge.json"


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def convert_conversation(conversation: Sequence[dict]) -> List[dict]:
    role_map = {"user": "human", "assistant": "gpt"}
    converted: List[dict] = []
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        converted.append({
            "from": role_map.get(role, role),
            "value": content,
        })
    return converted


def convert_records(records: List[dict]) -> List[dict]:
    dataset: List[dict] = []
    for rec in records:
        dataset.append({
            "conversations": convert_conversation(rec["conversation"]),
            "tools": "",
            "system": rec["system"],
        })
    return dataset


def write_json(data: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(data)} items to {path}")


def main() -> None:
    all_datasets: List[List[dict]] = []

    for src in SOURCES:
        records = load_jsonl(src["path"])
        dataset = convert_records(records)
        write_json(dataset, src["output"])
        all_datasets.append(dataset)

    # merged
    merged: List[dict] = []
    for ds in all_datasets:
        merged.extend(ds)
    write_json(merged, MERGED_OUTPUT)


if __name__ == "__main__":
    main()
