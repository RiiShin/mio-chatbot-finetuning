import csv
import json
import os
import re
import time

from openai import OpenAI
from tqdm import tqdm


# -----------------------------
# User settings
# -----------------------------
API_KEY = os.environ.get("OPENAI_API_KEY", "")  # Set via environment variable
MODEL = "gpt-5.2-2025-12-11"
USE_TEMPERATURE = False
TEMPERATURE = 0.7
EXTENSION_COUNT = 10

BATCH_COMPLETION_WINDOW = "24h"
BATCH_POLL_SECONDS = 10

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
BATCH_WORK_DIR = os.path.join(_PROJECT_ROOT, "batch_work")

SETTING_PATH = os.path.join(_PROJECT_ROOT, "card_sets", "基础setting.txt")
SEPARATOR = "⫶"

PROMPT_TEMPLATE = """你是训练数据编剧，要为单设定中文角色扮演聊天机器人生成 mio_set 的二级卡片（L2），用于放进prompt里的结构化变量，不是对话台词。
硬性要求：全中文；允许极少量日语固有名词（地名或店名）；不得出现“作为AI/系统/模型”等元叙述；不得输出可定位隐私；不得写成第一人称对话口吻；不得与给定设定全文冲突。
格式要求：严禁输出任何引号字符（" ' “ ” 「 」 『 』 等）；严禁出现英文逗号,；每条卡必须单行且不包含换行符；分隔符固定为⫶。

【给定设定全文（唯一权威，不可改动）】
{setting_txt}

输入：
- L1 card（本批次侧重点）：{l1_card}
- 生成数量：{n}

输出格式（非常重要）：
必须恰好输出{n}行，每行以“卡01：/卡02：.../卡{n}：”开头（序号连续），并且每行严格包含两段：事实段⫶recall_form
- 事实段：只写澪侧事实类设定，第三人称或中性描述；用“中文键=值”的紧凑格式，字段之间用英文分号;分隔；至少包含2个可回指事实点；禁止叙事与情绪长描写；禁止使用任何英文键。
  中文键示例（仅示例）：乐队名、成员分工、成员昵称、排练地点、原创曲名、曲目主题、节奏BPM、和弦走向、DEMO文件名、校园祭歌单、翻唱艺人、口头禅、私物细节、小目标进度、时间表变动。
- recall_form：只允许一行自然语言，格式必须是：被问到{{触发主题}}时回指{{字段1}} 或 被问到{{触发主题}}时回指{{字段1}}、{{字段2}}
  约束：最多回指2个字段；字段名必须严格等于事实段里的中文键名；触发主题要与字段对应；触发主题用1-3个短主题词用“/”连接；不得使用引号、不得使用英文逗号。

现在开始输出：只输出卡片本体，不要任何解释或标题。
"""


INPUT_DIR = os.path.join(_PROJECT_ROOT, "card_sets", "l1", "csv")
INPUT_CSV_NAME = "mio_set_l1.csv"
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "card_sets", "l2", "csv")
OUTPUT_CSV_NAME = "mio_set_l2.csv"
INPUT_ID_FIELD = "id"
INPUT_TEXT_FIELD = "card content"
OUTPUT_HEADER = ["id_1", "id_2", "extend_card", "recall_form"]


def resolve_input_csv_path(input_dir: str, input_name: str) -> str:
    if input_name:
        return os.path.join(input_dir, input_name)

    csv_files = [
        name for name in os.listdir(input_dir)
        if name.lower().endswith(".csv")
    ]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .csv files found in {input_dir}")
    if len(csv_files) > 1:
        raise ValueError(
            "Multiple .csv files found. Set INPUT_CSV_NAME to choose one."
        )
    return os.path.join(input_dir, csv_files[0])


def build_output_csv_path(input_csv_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    return os.path.join(output_dir, f"{OUTPUT_CSV_NAME}")


def build_request_jsonl_path(input_csv_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    return os.path.join(BATCH_WORK_DIR, f"{base_name}_requests.jsonl")


def build_error_report_path(input_csv_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    return os.path.join(BATCH_WORK_DIR, f"{base_name}_batch_errors.json")


def read_source_rows(input_csv_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            card_id = str(row.get(INPUT_ID_FIELD, "")).strip()
            card_text = str(row.get(INPUT_TEXT_FIELD, "")).strip()
            if not card_id or not card_text:
                continue
            rows.append({"id": card_id, "text": card_text})
    if not rows:
        raise ValueError("No valid rows found in input CSV.")
    return rows


def read_setting_text(setting_path: str) -> str:
    with open(setting_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def build_prompt(setting_text: str, l1_card: str) -> str:
    return PROMPT_TEMPLATE.format(
        setting_txt=setting_text,
        l1_card=l1_card,
        n=EXTENSION_COUNT,
        sep=SEPARATOR,
    )


def clean_extension_line(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^\s*卡\s*\d{1,3}\s*[：:]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*[-•]\s+", "", cleaned)
    cleaned = re.sub(r"^\s*\d{1,2}[.)\-:]\s+", "", cleaned)
    return cleaned.strip()


def split_extensions(raw_text: str) -> list[str]:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"卡\s*\d{1,3}\s*[：:]")
    matches = list(pattern.finditer(text))

    if len(matches) >= 2:
        segments: list[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            segment = text[start:end].strip()
            segment = re.sub(r"^\s*卡\s*\d{1,3}\s*[：:]\s*", "", segment)
            segment = segment.strip()
            if segment:
                segments.append(segment)
        return segments

    lines = [line for line in text.splitlines() if line.strip()]
    cleaned = [clean_extension_line(line) for line in lines]
    cleaned = [line for line in cleaned if line]
    return cleaned


def split_extension_parts(line: str, sep: str) -> tuple[str, str]:
    if sep in line:
        left, right = line.split(sep, 1)
        return left.strip(), right.strip()
    return line.strip(), ""


def build_batch_requests(
    rows: list[dict[str, str]],
    setting_text: str,
) -> tuple[list[dict[str, object]], list[str], dict[str, str]]:
    requests: list[dict[str, object]] = []
    custom_id_order: list[str] = []
    custom_to_original: dict[str, str] = {}

    for idx, row in enumerate(
        tqdm(rows, total=len(rows), desc="Preparing batch")
    ):
        custom_id = f"{row['id']}__{idx}"
        custom_id_order.append(custom_id)
        custom_to_original[custom_id] = row["id"]

        body: dict[str, object] = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": build_prompt(setting_text, row["text"]),
                },
            ],
        }
        if USE_TEMPERATURE:
            body["temperature"] = TEMPERATURE

        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    return requests, custom_id_order, custom_to_original


def write_jsonl(file_path: str, items: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def submit_batch(client: OpenAI, jsonl_path: str) -> tuple[str, str]:
    with open(jsonl_path, "rb") as file:
        input_file = client.files.create(file=file, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=BATCH_COMPLETION_WINDOW,
    )
    return batch.id, input_file.id


def wait_for_batch(client: OpenAI, batch_id: str) -> object:
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in {"completed", "failed", "expired", "cancelled"}:
            return batch
        print(batch.status, getattr(batch, "request_counts", None))
        time.sleep(BATCH_POLL_SECONDS)


def download_file_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)

    if hasattr(content, "text"):
        return content.text

    if hasattr(content, "read"):
        data = content.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")

    return str(content)


def parse_batch_output(
    output_text: str,
    custom_id_order: list[str],
    custom_to_original: dict[str, str],
) -> tuple[
    dict[str, list[str]],
    list[dict[str, object]],
    list[str],
    list[dict[str, object]],
]:
    custom_to_extensions: dict[str, list[str]] = {}
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    lines = output_text.splitlines()
    for line in tqdm(lines, total=len(lines), desc="Parsing batch output"):
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        if custom_id not in custom_to_original:
            continue

        error = record.get("error")
        if error:
            errors.append({"custom_id": custom_id, "error": error})
            continue

        response = record.get("response")
        if not response:
            errors.append({"custom_id": custom_id, "error": "missing response"})
            continue

        status = response.get("status_code")
        if status != 200:
            errors.append({
                "custom_id": custom_id,
                "error": f"status_code={status}",
            })
            continue

        body = response.get("body", {})
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            content = json.dumps(body, ensure_ascii=False)

        content_text = str(content)
        extensions = split_extensions(content_text)

        if len(extensions) == 0:
            errors.append({
                "custom_id": custom_id,
                "error": "no valid extensions parsed",
                "content_preview": content_text[:300],
            })
            continue

        issues: list[str] = []
        if len(extensions) != EXTENSION_COUNT:
            issues.append(
                f"expected {EXTENSION_COUNT} lines, got {len(extensions)}"
            )

        missing_separator = [
            index + 1
            for index, item in enumerate(extensions)
            if SEPARATOR not in item
        ]
        if missing_separator:
            issues.append(
                f"missing separator on lines: {missing_separator[:8]}"
            )

        if issues:
            errors.append({
                "custom_id": custom_id,
                "error": "; ".join(issues),
                "content_preview": content_text[:300],
                "parsed_preview": extensions[:3],
            })
            continue

        custom_to_extensions[custom_id] = extensions

    missing = [
        custom_id for custom_id in custom_id_order
        if custom_id not in custom_to_extensions
    ]
    return custom_to_extensions, errors, missing, warnings


def build_output_rows(
    custom_id_order: list[str],
    custom_to_original: dict[str, str],
    custom_to_extensions: dict[str, list[str]],
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for custom_id in custom_id_order:
        extensions = custom_to_extensions.get(custom_id)
        if not extensions:
            continue

        original_id = custom_to_original[custom_id]
        for i, ext in enumerate(extensions):
            fact_part, recall_part = split_extension_parts(ext, SEPARATOR)
            output_rows.append({
                "id_1": original_id,
                "id_2": str(i),
                "extend_card": fact_part,
                "recall_form": recall_part,
            })
    return output_rows


def write_error_report(
    error_path: str,
    errors: list[dict[str, object]],
    warnings: list[dict[str, object]],
    missing: list[str],
) -> None:
    os.makedirs(os.path.dirname(error_path), exist_ok=True)
    with open(error_path, "w", encoding="utf-8") as file:
        json.dump(
            {"errors": errors, "warnings": warnings, "missing_custom_ids": missing},
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_extended_csv(output_csv_path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(OUTPUT_HEADER)
        for row in rows:
            writer.writerow([
                row["id_1"],
                row["id_2"],
                row["extend_card"],
                row["recall_form"],
            ])


def main() -> None:
    if not API_KEY.strip():
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    input_csv_path = resolve_input_csv_path(INPUT_DIR, INPUT_CSV_NAME)
    output_csv_path = build_output_csv_path(input_csv_path, OUTPUT_DIR)
    request_jsonl_path = build_request_jsonl_path(input_csv_path)
    error_report_path = build_error_report_path(input_csv_path)

    setting_text = read_setting_text(SETTING_PATH)
    source_rows = read_source_rows(input_csv_path)
    requests, custom_id_order, custom_to_original = build_batch_requests(
        source_rows,
        setting_text,
    )
    write_jsonl(request_jsonl_path, requests)

    client = OpenAI(api_key=API_KEY)
    batch_id, input_file_id = submit_batch(client, request_jsonl_path)
    print(f"Batch submitted: {batch_id} (input file: {input_file_id})")

    batch = wait_for_batch(client, batch_id)
    status = getattr(batch, "status", None)
    if status != "completed":
        raise RuntimeError(f"Batch did not complete: {status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("Batch completed without output_file_id.")

    output_text = download_file_text(client, output_file_id)
    custom_to_extensions, errors, missing, warnings = parse_batch_output(
        output_text=output_text,
        custom_id_order=custom_id_order,
        custom_to_original=custom_to_original,
    )

    if errors or missing:
        write_error_report(error_report_path, errors, warnings, missing)
        raise RuntimeError(
            f"Batch completed with errors. See: {error_report_path}"
        )

    output_rows = build_output_rows(
        custom_id_order=custom_id_order,
        custom_to_original=custom_to_original,
        custom_to_extensions=custom_to_extensions,
    )
    write_extended_csv(output_csv_path, output_rows)
    print(f"Saved: {output_csv_path}")


if __name__ == "__main__":
    main()
