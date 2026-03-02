import csv
import os


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
TXT_FILE_PATH = os.path.join(_PROJECT_ROOT, "card_sets", "l1", "original", "user_want.txt")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "card_sets", "l1", "csv")
CSV_HEADER = ["id", "card content"]


def read_lines_from_txt(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as txt_file:
        return [line.rstrip("\n").rstrip("\r") for line in txt_file]


def build_output_csv_path(txt_path: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    return os.path.join(output_dir, f"{base_name}.csv")


def write_string_list_to_csv(
    output_path: str,
    header: list[str],
    string_list: list[str],
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for index, value in enumerate(string_list):
            writer.writerow([index, value])


if __name__ == "__main__":
    lines = read_lines_from_txt(TXT_FILE_PATH)
    output_csv_path = build_output_csv_path(TXT_FILE_PATH, OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_string_list_to_csv(output_csv_path, CSV_HEADER, lines)
