from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class MemoryStore:
    """
    本地长期记忆存储。

    设计目标：
    1. 用 JSON 文件保存长期记忆
    2. 支持按 content 去重追加
    3. 支持按 key 更新同一类记忆
    4. 支持把长期记忆格式化成 prompt
    """

    path: Path
    memories: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryStore":
        memory_path = Path(path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)

        if not memory_path.exists():
            return cls(path=memory_path, memories=[])

        try:
            data = json.loads(memory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"长期记忆文件不是合法 JSON：{memory_path}") from exc

        memories = data.get("memories", [])
        if not isinstance(memories, list):
            memories = []

        normalized_memories: list[dict[str, Any]] = []
        for item in memories:
            if isinstance(item, dict):
                normalized_memories.append(item)

        return cls(path=memory_path, memories=normalized_memories)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "memories": self.memories,
        }

        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_memory(
        self,
        memory_type: str,
        content: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> bool:
        clean_content = content.strip()
        if not clean_content:
            return False

        clean_tags = tags or []

        for memory in self.memories:
            if memory.get("content") == clean_content:
                memory["type"] = memory_type
                memory["importance"] = importance
                memory["tags"] = clean_tags
                memory["source"] = source
                memory["updated_at"] = _now_iso()
                self.save()
                return False

        now = _now_iso()
        self.memories.append(
            {
                "type": memory_type,
                "content": clean_content,
                "importance": importance,
                "tags": clean_tags,
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
        )

        self.save()
        return True

    def upsert_memory(
        self,
        key: str,
        memory_type: str,
        content: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> str:
        clean_key = key.strip()
        clean_content = content.strip()

        if not clean_key or not clean_content:
            return "skipped"

        clean_tags = tags or []

        for memory in self.memories:
            if memory.get("key") == clean_key:
                memory["type"] = memory_type
                memory["content"] = clean_content
                memory["importance"] = importance
                memory["tags"] = clean_tags
                memory["source"] = source
                memory["updated_at"] = _now_iso()
                self.save()
                return "updated"

        now = _now_iso()
        self.memories.append(
            {
                "key": clean_key,
                "type": memory_type,
                "content": clean_content,
                "importance": importance,
                "tags": clean_tags,
                "source": source,
                "created_at": now,
                "updated_at": now,
            }
        )

        self.save()
        return "added"

    def format_for_prompt(self, limit: int = 20) -> str:
        if not self.memories:
            return "暂无长期记忆。"

        sorted_memories = sorted(
            self.memories,
            key=lambda item: (
                float(item.get("importance", 0.0)),
                str(item.get("updated_at", "")),
            ),
            reverse=True,
        )

        lines: list[str] = []
        for index, memory in enumerate(sorted_memories[:limit], start=1):
            memory_type = str(memory.get("type", "fact"))
            content = str(memory.get("content", "")).strip()
            importance = memory.get("importance", 0.0)
            key = str(memory.get("key", "")).strip()
            tags = memory.get("tags", [])

            if not content:
                continue

            tag_text = ""
            if isinstance(tags, list) and tags:
                tag_text = " | tags: " + ", ".join(str(tag) for tag in tags)

            key_text = f" | key: {key}" if key else ""

            lines.append(
                f"{index}. [{memory_type}] {content} "
                f"(importance: {importance}{key_text}{tag_text})"
            )

        return "\n".join(lines) if lines else "暂无长期记忆。"
