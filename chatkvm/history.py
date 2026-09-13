"""一场对话一个目录：进行中写 history/latest，归档为 history/时间戳。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path("history")
JSONL = "messages.jsonl"


class History:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self._jsonl = self.folder / JSONL

    @classmethod
    def current(cls) -> History:
        return cls(ROOT / "latest")

    @classmethod
    def rotate(cls) -> History:
        latest = ROOT / "latest"
        jsonl = latest / JSONL
        if jsonl.exists() and jsonl.read_text(encoding="utf-8").strip():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            dest = ROOT / stamp
            suffix = 2
            while dest.exists():
                dest = ROOT / f"{stamp}-{suffix}"
                suffix += 1
            latest.rename(dest)
        return cls(ROOT / "latest")

    def load(self) -> list[dict[str, Any]]:
        if not self._jsonl.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self._jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def append(
        self,
        role: str,
        *,
        started: float | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "role": role,
            **{key: value for key, value in fields.items() if value is not None},
        }
        item["at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if started is not None:
            item["ms"] = max(int((time.monotonic() - started) * 1000), 0)
        with self._jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        return item

    def save_image(self, role: str, src: Path) -> str:
        dest_dir = self.folder / role
        dest_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        while (dest_dir / f"{n:04d}.jpg").exists():
            n += 1
        dest = dest_dir / f"{n:04d}.jpg"
        dest.write_bytes(Path(src).read_bytes())
        return f"{role}/{dest.name}"

    def resolve(self, image: str) -> Path:
        return self.folder / image

    def ollama_messages(self, system: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        last_screen: str | None = None
        for item in self.load():
            role = item.get("role")
            if role == "screen":
                if item.get("image"):
                    last_screen = str(item["image"])
                continue
            if role not in {"user", "assistant", "tool"}:
                continue
            row: dict[str, Any] = {"role": role, "content": item.get("content") or ""}
            if role == "assistant" and item.get("tool_calls"):
                row["tool_calls"] = item["tool_calls"]
            if role == "tool" and item.get("tool_name"):
                row["tool_name"] = item["tool_name"]
            out.append(row)
        if last_screen:
            out.append(
                {
                    "role": "user",
                    "content": "当前画面：",
                    "images": [str(self.resolve(last_screen))],
                }
            )
        return out
