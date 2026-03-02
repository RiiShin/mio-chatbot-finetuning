#!/usr/bin/env python3
"""gen_outlines.py  —  Plan B Step 1: Conversation outline generation.

For each conversation, samples:
  1 user_profile  +  1 scene  +  optionally 1 refusal
then asks an LLM to write a free-form conversation outline/arc.

Uses the OpenAI Batch API (24h window, 50% cheaper).

Output: one JSONL file where each line is:
{
  "conv_id": int,
  "num_turns": int,
  "user_profile_id": int,  "user_profile": str,
  "scene_id": int,          "scene": str,
  "has_refusal": bool,
  "refusal_id": int|null,   "refusal": str|null,
  "outline": str            // generated outline text
}

Usage example:
  export OPENAI_API_KEY="sk-..."
  python gen_outlines.py --num-conversations 350 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_COMPLETION_WINDOW = "24h"
BATCH_POLL_SECONDS = 30
DEFAULT_MODEL = "gpt-5.1-2025-11-13"
# gpt-5.1-2025-11-13
# gpt-5-mini-2025-08-07
# gpt-5-2025-08-07

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
CARD_DIR = os.path.join(_PROJECT_ROOT, "card_sets")
DEFAULT_USER_PROFILE_CSV = os.path.join(CARD_DIR, "l1", "csv", "user_profiles.csv")
DEFAULT_SCENE_CSV = os.path.join(CARD_DIR, "l1", "csv", "scene.csv")
DEFAULT_REFUSAL_CSV = os.path.join(CARD_DIR, "l1", "csv", "refusal.csv")
DEFAULT_SETTING_TXT = os.path.join(CARD_DIR, "腹黑setting.txt")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outlines")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Card:
    card_id: str
    content: str


def load_cards(path: str) -> List[Card]:
    """Load a CSV with columns (id, card content) into a list of Card."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    cards: List[Card] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            card_id = (row.get("id") or "").strip()
            content = (row.get("card content") or "").strip()
            if card_id and content:
                cards.append(Card(card_id=card_id, content=content))
    if not cards:
        raise ValueError(f"No cards loaded from {path}")
    return cards


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Character summary (condensed from full setting, for outline prompt only)
# ---------------------------------------------------------------------------
CHARACTER_SUMMARY = """\
星川澪（17岁），东京都立樱坂高等学校高三，轻音部部长，乐队「樱坂Spica」主唱兼吉他手。

性格：腹黑毒舌系——嘴上从不饶人，但行动上永远第一个帮忙。
- 嘴硬心软：夸人要拐弯（"还行吧""不算太差"），帮人要找借口（"才不是为了你，只是刚好顺路"）
- 调侃不越线：吐槽行为不攻击人格，嘲讽完一定跟上有用的东西或温柔的补刀
- 反问狂人：习惯用反问代替直接回答，制造互动感
- 音乐认真模式：一聊到吉他、乐队、唱歌，突然变得专注且热情，毒舌浓度降低
- 真情泄露立刻找补：不小心说了暖心话会马上否认（"别误会啊""又不是特意的"）

乐队成员：るな（贝斯，天然呆死党）、ましろ（鼓手，热血笨蛋）、ゆい（键盘，沉默毒舌）
代表曲：《星屑ノイズ》《走り出す理由》
喜欢：J-Rock、后摇、动画歌曲、凛として時雨、AKFG、Aimer、LiSA、YOASOBI
日常：骑车上学、放学买草莓布丁和冰奶茶、排练到18:30、晚上练琴写歌词、听Aimer入睡
不聊：政治、投资、编程、成人内容；不给真实联系方式/地址"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_outline_prompt(
    user_profile: Card,
    scene: Card,
    num_turns: int,
    refusal: Optional[Card],
) -> str:
    """Build the user-message prompt for outline generation."""

    refusal_block = ""
    refusal_rule = ""
    if refusal:
        refusal_block = (
            f"\n【拒绝事件（本次对话必须包含一次）】\n"
            f"{refusal.content}\n"
        )
        refusal_rule = (
            "5. 对话中需自然出现一次用户越界请求（参见【拒绝事件】），"
            "澪在角色内拒绝后把话题拉回正轨。越界请求不要放在开头或结尾。\n"
        )

    prompt = f"""\
请为一段角色扮演聊天设计对话大纲。

【角色】
{CHARACTER_SUMMARY}

【用户画像】
{user_profile.content}

【场景氛围】
{scene.content}

【对话长度】{num_turns}轮（每轮 = 用户发1条 + 澪回1条）
{refusal_block}
请设计这段 {num_turns} 轮对话的大纲。要求：
1. 用自然语言描述对话的流向和情绪弧线（开场 → 发展 → 转折/高潮 → 收尾），不要写"第X轮"的逐轮分配
2. 用户的行为要贴合画像特征（话多/话少、主动/被动、当前情绪状态等）
3. 标注哪些时刻适合展现澪的腹黑特征（吐槽、嘴硬心软、音乐认真模式、真情泄露找补等）
4. 整体要有起承转合和情绪波动，不要全程同一个温度
{refusal_rule}
输出要求：
- 3-4段自然语言，总计400-600字
- 描述情绪走向和关键转折点，不要写角色的具体台词或对白
- 不要使用"第X轮""X-Y轮"等轮次编号
- 不要用Markdown格式"""

    return prompt


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_conversations(
    user_profiles: List[Card],
    scenes: List[Card],
    refusals: List[Card],
    num_conversations: int,
    p_refusal: float,
    turn_min: int,
    turn_max: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    for conv_id in range(1, num_conversations + 1):
        profile = rng.choice(user_profiles)
        scene = rng.choice(scenes)
        num_turns = rng.randint(turn_min, turn_max)

        has_refusal = rng.random() < p_refusal
        refusal = rng.choice(refusals) if has_refusal else None

        samples.append({
            "conv_id": conv_id,
            "num_turns": num_turns,
            "user_profile": profile,
            "scene": scene,
            "has_refusal": has_refusal,
            "refusal": refusal,
        })

    return samples


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------
def build_batch_requests(
    samples: List[Dict[str, Any]],
    model: str,
    temperature: Optional[float],
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []

    for sample in samples:
        prompt = build_outline_prompt(
            user_profile=sample["user_profile"],
            scene=sample["scene"],
            num_turns=sample["num_turns"],
            refusal=sample["refusal"],
        )

        custom_id = f"outline_{sample['conv_id']}"

        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是专业的对话大纲设计师。"
                        "根据给定的角色设定、用户画像和场景，"
                        "设计自然流畅、有情绪弧线的对话大纲。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 2048,
        }
        if temperature is not None:
            body["temperature"] = temperature

        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    return requests


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def submit_batch(client: OpenAI, jsonl_path: str) -> Tuple[str, str]:
    with open(jsonl_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
    )
    return batch.id, input_file.id


def wait_for_batch(client: OpenAI, batch_id: str) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)

        if status in {"completed", "failed", "expired", "cancelled"}:
            print(f"  Batch finished: {status} | {counts}")
            return batch

        print(f"  Polling… status={status} | {counts}")
        time.sleep(BATCH_POLL_SECONDS)


def download_file_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    return content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------
def parse_batch_results(
    output_text: str,
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse batch output, merge with sample metadata, return (results, errors)."""
    sample_by_id = {f"outline_{s['conv_id']}": s for s in samples}

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for line in output_text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        sample = sample_by_id.get(custom_id)
        if not sample:
            continue

        # Check for API-level error
        error = record.get("error")
        if error:
            errors.append({"custom_id": custom_id, "error": error})
            continue

        response = record.get("response", {})
        if response.get("status_code") != 200:
            errors.append({
                "custom_id": custom_id,
                "error": f"status_code={response.get('status_code')}",
            })
            continue

        try:
            outline_text = response["body"]["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            errors.append({"custom_id": custom_id, "error": "missing content in response"})
            continue

        # Minimal quality check: outline should be at least ~50 chars
        if len(outline_text) < 50:
            errors.append({
                "custom_id": custom_id,
                "error": f"outline too short ({len(outline_text)} chars)",
                "content": outline_text,
            })
            continue

        entry = {
            "conv_id": sample["conv_id"],
            "num_turns": sample["num_turns"],
            "user_profile_id": int(sample["user_profile"].card_id),
            "user_profile": sample["user_profile"].content,
            "scene_id": int(sample["scene"].card_id),
            "scene": sample["scene"].content,
            "has_refusal": sample["has_refusal"],
            "refusal_id": int(sample["refusal"].card_id) if sample["refusal"] else None,
            "refusal": sample["refusal"].content if sample["refusal"] else None,
            "outline": outline_text,
        }
        results.append(entry)

    results.sort(key=lambda x: x["conv_id"])
    return results, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan B Step 1: generate conversation outlines via OpenAI Batch API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num-conversations", type=int, default=350,
                        help="Total number of outlines to generate")
    parser.add_argument("--turn-min", type=int, default=10,
                        help="Minimum turns per conversation")
    parser.add_argument("--turn-max", type=int, default=15,
                        help="Maximum turns per conversation")
    parser.add_argument("--p-refusal", type=float, default=0.3,
                        help="Probability a conversation includes a refusal event")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (omit for models like gpt-5/gpt-5-mini that don't need it)")

    parser.add_argument("--user-profile-csv", type=str, default=DEFAULT_USER_PROFILE_CSV)
    parser.add_argument("--scene-csv", type=str, default=DEFAULT_SCENE_CSV)
    parser.add_argument("--refusal-csv", type=str, default=DEFAULT_REFUSAL_CSV)
    parser.add_argument("--setting-txt", type=str, default=DEFAULT_SETTING_TXT)

    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    # --- API key ---
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Provide --api-key or set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    # --- Load cards ---
    user_profiles = load_cards(args.user_profile_csv)
    scenes = load_cards(args.scene_csv)
    refusals = load_cards(args.refusal_csv)
    print(f"Loaded cards: {len(user_profiles)} user_profiles, "
          f"{len(scenes)} scenes, {len(refusals)} refusals")

    # --- Sample ---
    samples = sample_conversations(
        user_profiles=user_profiles,
        scenes=scenes,
        refusals=refusals,
        num_conversations=args.num_conversations,
        p_refusal=args.p_refusal,
        turn_min=args.turn_min,
        turn_max=args.turn_max,
        seed=args.seed,
    )
    refusal_count = sum(1 for s in samples if s["has_refusal"])
    print(f"Sampled {len(samples)} conversations ({refusal_count} with refusal)")

    # --- Build batch requests ---
    requests = build_batch_requests(
        samples=samples,
        model=args.model,
        temperature=args.temperature,
    )

    # Show first prompt for inspection
    print("\n" + "=" * 60)
    print("FIRST PROMPT (for inspection):")
    print("=" * 60)
    print(requests[0]["body"]["messages"][1]["content"])
    print("=" * 60 + "\n")

    # --- Write request JSONL ---
    tag = f"n{args.num_conversations}_s{args.seed}"
    work_dir = os.path.join(args.output_dir, "batch_work")
    request_path = os.path.join(work_dir, f"{tag}_requests.jsonl")
    write_jsonl(request_path, requests)
    print(f"Request JSONL written: {request_path} ({len(requests)} requests)")

    # --- Submit batch ---
    client = OpenAI(api_key=api_key)
    batch_id, input_file_id = submit_batch(client, request_path)
    print(f"Batch submitted: id={batch_id}, input_file={input_file_id}")
    print("Waiting for batch to complete (this may take a while)…\n")

    # --- Poll ---
    batch = wait_for_batch(client, batch_id)
    if batch.status != "completed":
        print(f"\nERROR: Batch ended with status '{batch.status}'", file=sys.stderr)
        err_file_id = getattr(batch, "error_file_id", None)
        if err_file_id:
            err_text = download_file_text(client, err_file_id)
            err_path = os.path.join(work_dir, f"{tag}_batch_error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            print(f"  Error details saved to {err_path}")
        sys.exit(1)

    # --- Check for per-request errors (error_file_id) ---
    err_file_id = getattr(batch, "error_file_id", None)
    if err_file_id:
        err_text = download_file_text(client, err_file_id)
        err_path = os.path.join(work_dir, f"{tag}_request_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err_text)
        # Show first error for quick diagnosis
        first_err_line = err_text.strip().split("\n")[0] if err_text.strip() else ""
        if first_err_line:
            try:
                first_err = json.loads(first_err_line)
                err_detail = first_err.get("response", {}).get("body", {}).get("error", first_err.get("error"))
                print(f"  [REQUEST ERROR] {json.dumps(err_detail, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"  [REQUEST ERROR] {first_err_line[:300]}")
        print(f"  Full error log → {err_path}")

    # --- Download results ---
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        print("ERROR: Batch completed but no output_file_id (all requests may have failed).",
              file=sys.stderr)
        sys.exit(1)

    print("Downloading batch output…")
    output_text = download_file_text(client, output_file_id)

    # Save raw output for debugging
    raw_path = os.path.join(work_dir, f"{tag}_raw_output.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"Raw output saved: {raw_path}")

    # --- Parse ---
    results, errors = parse_batch_results(output_text, samples)

    if errors:
        err_path = os.path.join(work_dir, f"{tag}_parse_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"  {len(errors)} errors → {err_path}")

    # --- Write final outlines JSONL ---
    out_path = os.path.join(args.output_dir, f"outlines_{tag}.jsonl")
    write_jsonl(out_path, results)
    print(f"\nDone: {len(results)}/{args.num_conversations} outlines → {out_path}")

    # Show first result
    if results:
        print("\n" + "=" * 60)
        print("FIRST OUTLINE (for inspection):")
        print("=" * 60)
        print(json.dumps(results[0], ensure_ascii=False, indent=2))
        print("=" * 60)


if __name__ == "__main__":
    main()
