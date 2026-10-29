
import logging
import os

from project_code.app_config.settings import app_settings


def _flatten_for_log(data: dict) -> list[str]:
    lines = []
    for k, v in (data or {}).items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for i, item in enumerate(v):
                lines.append(f"  [{i}] {item}")
        elif isinstance(v, dict):
            lines.append(f"{k}:")
            for sub_k, sub_v in item.items() if False else v.items():
                lines.append(f"  {sub_k}: {sub_v}")
        else:
            lines.append(f"{k}: {v}")
    return lines


def log_boxed(title: str, lines: list[str]):
    content = [title] + lines
    width = max(len(s) for s in content)
    log_lines = ["┌" + "─" * (width + 2) + "┐"]
    for s in content:
        log_lines.append(f"│ {s:<{width}} │")
    log_lines.append("└" + "─" * (width + 2) + "┘")
    logging.info("\n" + "\n".join(log_lines))

def create_output_folder():
    folder_path = os.path.join(app_settings.logging_and_records.output_path, CURRENT_DATE)
    os.makedirs(folder_path, exist_ok=True)
