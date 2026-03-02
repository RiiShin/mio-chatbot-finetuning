#!/usr/bin/env python3
"""gen_conversations.py  —  Plan B Step 2: Generate full conversations from outlines.

Reads the outlines JSONL produced by gen_outlines.py, then for each outline
asks the LLM to generate a complete multi-turn conversation.

Key design:
  - System prompt embeds the FULL character card (腹黑setting.txt) so the model
    internalises Mio's voice, speech patterns, and few-shot examples.
  - User prompt provides the outline + user profile + scene + minimal formatting rules.
  - The LLM outputs a JSON array of alternating user/assistant messages.
  - The saved "system" field is the pure character card (for downstream SFT),
    not the generation-specific framing.

Uses the OpenAI Batch API.

Output JSONL — each line:
{
  "conv_id": int,
  "num_turns": int,
  "user_profile_id": int,  "user_profile": str,
  "scene_id": int,          "scene": str,
  "has_refusal": bool,
  "refusal_id": int|null,   "refusal": str|null,
  "system": str,             // pure character card for SFT
  "conversation": [{"role":"user","content":"..."},{"role":"assistant","content":"..."},...]
}

Usage:
  python gen_conversations.py --outlines /path/to/outlines_n350_s42.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_COMPLETION_WINDOW = "24h"
BATCH_POLL_SECONDS = 30
DEFAULT_MODEL = "gpt-5.1-2025-11-13"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
DEFAULT_SETTING_TXT = os.path.join(_PROJECT_ROOT, "card_sets", "腹黑setting.txt")
DEFAULT_SFT_SYSTEM_TXT = os.path.join(_PROJECT_ROOT, "sft_system_prompt.txt")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "v2_convs")
# gpt-5.1-2025-11-13
# gpt-5-mini-2025-08-07
# gpt-5-2025-08-07

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_outlines(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"No outlines loaded from {path}")
    return entries


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_generation_system_prompt(character_setting: str) -> str:
    """System prompt for the generation call.

    Embeds the full character card so the model deeply internalises Mio's
    personality, speech patterns, and the few-shot reply examples.
    """
    return (
        "你需要根据用户提供的对话大纲，生成一段完整的角色扮演对话。\n"
        "在这段对话中，assistant 就是以下角色——"
        "请完全以她的口吻、性格和说话方式来写 assistant 的每一条回复。\n\n"
        "===== 角色设定 =====\n"
        f"{character_setting}\n"
        "===== 角色设定结束 =====\n\n"
        "核心要求：\n"
        "- assistant 的每条回复必须以 #[动作][表情] 标签开头（英文小写），"
        "例如 #[arms crossed][smirk]\n"
        "- assistant 回复要短，1-3句，像真实聊天节奏\n"
        "- assistant 必须完全贴合上面的角色性格和说话风格，"
        "尤其是腹黑毒舌、嘴硬心软的特征\n"
        "- 只输出纯 JSON 数组，不要任何其他文字或 Markdown 标记"
    )


def build_generation_user_prompt(entry: Dict[str, Any]) -> str:
    """User-message prompt for a single conversation."""
    num_turns = entry["num_turns"]
    expected_len = num_turns * 2

    refusal_note = ""
    if entry.get("has_refusal") and entry.get("refusal"):
        refusal_note = (
            f"\n【拒绝事件】对话中需包含一次用户越界请求，澪需在角色内拒绝：\n"
            f"{entry['refusal']}\n"
        )

    return (
        f"请根据以下大纲生成完整对话。\n\n"
        f"【对话大纲】\n{entry['outline']}\n\n"
        f"【用户画像】\n{entry['user_profile']}\n\n"
        f"【场景氛围】\n{entry['scene']}\n"
        f"{refusal_note}\n"
        f"【输出格式】\n"
        f"- JSON 数组，长度恰好 {expected_len}（{num_turns}轮，每轮 user + assistant）\n"
        f"- 严格交替：user, assistant, user, assistant...\n"
        f"- 格式：{{\"role\":\"user\",\"content\":\"...\"}} 或 "
        f"{{\"role\":\"assistant\",\"content\":\"...\"}}\n\n"
        f"【user 消息硬性约束】\n"
        f"- 每条 user 消息必须短，15-50字（偶尔60字），像真实即时通讯聊天\n"
        f"- user 是 ego-centric 的：主要说自己的事、自己的感受、自己的困扰，不会长篇分析澪或音乐话题\n"
        f"- 话多/话少、主动/被动都要贴合用户画像；但即使「话多」，也是多发几条短消息，不是一条说一大段\n"
        f"- 不要以\"好的/嗯嗯/谢谢/明白了\"开头；不要机械重复澪说的话；不要过度配合澪的每个梗\n"
        f"- 高中生就抱怨月考作业、社畜就说加班累、宅就说动漫游戏——贴合身份的具体日常，不要泛泛而谈\n\n"
        f"【assistant 消息要求】\n"
        f"- 遵循大纲的情绪弧线和节奏，但不要把大纲原文写进对话\n\n"
        f"现在输出 JSON 数组："
    )


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------
def parse_conversation(content: str, num_turns: int) -> List[Dict[str, str]]:
    """Parse the generated JSON array and do basic validation."""
    content = content.strip()

    # Strip markdown fences
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()

    data = json.loads(content)

    # Unwrap if the model wrapped it in a dict
    if isinstance(data, dict):
        for key in ("data", "messages", "conversation", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Output is not a JSON array")

    expected_len = num_turns * 2

    # Truncate if too long
    if len(data) > expected_len:
        data = data[:expected_len]

    # Drop trailing odd element to keep pairs
    if len(data) % 2 == 1:
        data = data[:-1]

    # Allow up to 2 turns short
    min_len = expected_len - 4
    if len(data) < min_len:
        raise ValueError(f"Too short: got {len(data)}, need >= {min_len}")

    # Validate structure
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Turn {i} is not an object")
        role = item.get("role")
        text = str(item.get("content", "")).strip()
        if not text:
            raise ValueError(f"Turn {i} has empty content")
        expected_role = "user" if i % 2 == 0 else "assistant"
        if role != expected_role:
            raise ValueError(f"Turn {i}: expected {expected_role}, got {role}")

    return data


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan B Step 2: generate full conversations from outlines via Batch API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--outlines", type=str, required=True,
                        help="Path to outlines JSONL (output of gen_outlines.py)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (omit for gpt-5 family)")
    parser.add_argument("--setting-txt", type=str, default=DEFAULT_SETTING_TXT,
                        help="Full character card for generation system prompt")
    parser.add_argument("--sft-system-txt", type=str, default=DEFAULT_SFT_SYSTEM_TXT,
                        help="Condensed system prompt saved into output for SFT")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")

    args = parser.parse_args()

    # --- API key ---
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Provide --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    # --- Load ---
    character_setting = load_text(args.setting_txt)
    sft_system_prompt = load_text(args.sft_system_txt)
    outlines = load_outlines(args.outlines)
    print(f"Loaded {len(outlines)} outlines from {args.outlines}")
    print(f"SFT system prompt: {len(sft_system_prompt)} chars from {args.sft_system_txt}")

    # System prompts
    gen_system_prompt = build_generation_system_prompt(character_setting)

    # --- Build batch requests ---
    requests: List[Dict[str, Any]] = []
    for entry in outlines:
        custom_id = f"conv_{entry['conv_id']}"
        user_prompt = build_generation_user_prompt(entry)

        body: Dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": gen_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": 4096,
        }
        if args.temperature is not None:
            body["temperature"] = args.temperature

        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    # --- Inspect first prompt ---
    print(f"\n{'=' * 60}")
    print("GENERATION SYSTEM PROMPT (first 600 chars):")
    print(gen_system_prompt[:600] + "…")
    print(f"{'=' * 60}")
    print("FIRST USER PROMPT:")
    print(requests[0]["body"]["messages"][1]["content"])
    print(f"{'=' * 60}\n")

    # --- Write request JSONL ---
    outline_basename = os.path.splitext(os.path.basename(args.outlines))[0]
    tag = outline_basename.replace("outlines_", "") + "_shorter"
    work_dir = os.path.join(args.output_dir, "batch_work")
    request_path = os.path.join(work_dir, f"{tag}_conv_requests.jsonl")
    write_jsonl(request_path, requests)
    print(f"Wrote {len(requests)} requests → {request_path}")

    # --- Submit batch ---
    client = OpenAI(api_key=api_key)
    batch_id, input_file_id = submit_batch(client, request_path)
    print(f"Batch submitted: {batch_id}")
    print("Waiting for completion…\n")

    # --- Poll ---
    batch = wait_for_batch(client, batch_id)
    if batch.status != "completed":
        print(f"\nERROR: Batch ended with status '{batch.status}'", file=sys.stderr)
        err_fid = getattr(batch, "error_file_id", None)
        if err_fid:
            err_text = download_file_text(client, err_fid)
            err_path = os.path.join(work_dir, f"{tag}_conv_batch_error.txt")
            with open(err_path, "w", encoding="utf-8") as f:
                f.write(err_text)
            print(f"  Error details → {err_path}")
        sys.exit(1)

    # --- Check for per-request errors (error_file_id) ---
    err_file_id = getattr(batch, "error_file_id", None)
    if err_file_id:
        err_text = download_file_text(client, err_file_id)
        err_path = os.path.join(work_dir, f"{tag}_conv_request_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(err_text)
        first_err_line = err_text.strip().split("\n")[0] if err_text.strip() else ""
        if first_err_line:
            try:
                first_err = json.loads(first_err_line)
                err_detail = first_err.get("response", {}).get("body", {}).get("error", first_err.get("error"))
                print(f"  [REQUEST ERROR] {json.dumps(err_detail, ensure_ascii=False)}")
            except json.JSONDecodeError:
                print(f"  [REQUEST ERROR] {first_err_line[:300]}")
        print(f"  Full error log → {err_path}")

    # --- Download ---
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        print("ERROR: No output_file_id (all requests may have failed).", file=sys.stderr)
        sys.exit(1)

    print("Downloading batch output…")
    output_text = download_file_text(client, output_file_id)

    raw_path = os.path.join(work_dir, f"{tag}_conv_raw_output.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"Raw output saved → {raw_path}")

    # --- Parse results ---
    outline_by_id = {f"conv_{e['conv_id']}": e for e in outlines}
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for line in output_text.splitlines():
        if not line.strip():
            continue

        record = json.loads(line)
        custom_id = record.get("custom_id")
        entry = outline_by_id.get(custom_id)
        if not entry:
            continue

        # API-level error
        error = record.get("error")
        if error:
            errors.append({"custom_id": custom_id, "error": error})
            continue

        resp = record.get("response", {})
        if resp.get("status_code") != 200:
            errors.append({
                "custom_id": custom_id,
                "error": f"status_code={resp.get('status_code')}",
            })
            continue

        try:
            raw_content = resp["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            errors.append({"custom_id": custom_id, "error": "missing content"})
            continue

        try:
            conversation = parse_conversation(raw_content, entry["num_turns"])
        except (json.JSONDecodeError, ValueError) as e:
            errors.append({
                "custom_id": custom_id,
                "error": str(e),
                "content_preview": str(raw_content)[:300],
            })
            continue

        results.append({
            "conv_id": entry["conv_id"],
            "num_turns": entry["num_turns"],
            "user_profile_id": entry["user_profile_id"],
            "user_profile": entry["user_profile"],
            "scene_id": entry["scene_id"],
            "scene": entry["scene"],
            "has_refusal": entry["has_refusal"],
            "refusal_id": entry.get("refusal_id"),
            "refusal": entry.get("refusal"),
            "system": sft_system_prompt,
            "conversation": conversation,
        })

    results.sort(key=lambda x: x["conv_id"])

    # --- Error report ---
    if errors:
        err_path = os.path.join(work_dir, f"{tag}_conv_parse_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"  {len(errors)} errors → {err_path}")

    # --- Write final output ---
    out_path = os.path.join(args.output_dir, f"conversations_{tag}.jsonl")
    write_jsonl(out_path, results)
    print(f"\nDone: {len(results)}/{len(outlines)} conversations → {out_path}")

    # --- Show first result ---
    if results:
        first = results[0]
        print(f"\n{'=' * 60}")
        print(f"FIRST CONVERSATION (conv_id={first['conv_id']}, "
              f"{len(first['conversation'])} messages):")
        print(f"{'=' * 60}")
        for msg in first["conversation"][:6]:
            role = msg["role"]
            text = msg["content"]
            preview = text if len(text) <= 120 else text[:120] + "…"
            print(f"  [{role}] {preview}")
        if len(first["conversation"]) > 6:
            print(f"  … ({len(first['conversation']) - 6} more messages)")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
