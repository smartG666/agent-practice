from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# 是否开启 memory_store 调试日志
DEBUG = True


def debug_log(message: str) -> None:
    """
    打印 memory_store 模块调试日志。
    """
    if DEBUG:
        print(f"[DEBUG][memory_store] {message}")


def preview_text(text: str, limit: int = 80) -> str:
    """
    生成文本预览，避免日志太长。
    """
    return text.replace("\n", " ")[:limit]


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
        debug_log("准备加载长期记忆")
        debug_log(f"长期记忆文件路径：{memory_path}")

        memory_path.parent.mkdir(parents=True, exist_ok=True)

        if not memory_path.exists():
            debug_log("长期记忆文件不存在，将创建空 MemoryStore")
            return cls(path=memory_path, memories=[])

        debug_log("长期记忆文件存在，开始读取 JSON")

        try:
            data = json.loads(memory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            debug_log(f"长期记忆 JSON 解析失败：{exc}")
            raise RuntimeError(f"长期记忆文件不是合法 JSON：{memory_path}") from exc

        memories = data.get("memories", [])
        if not isinstance(memories, list):
            debug_log("memories 字段不是 list，已重置为空列表")
            memories = []

        normalized_memories: list[dict[str, Any]] = []
        skipped_count = 0

        for item in memories:
            if isinstance(item, dict):
                normalized_memories.append(item)
            else:
                skipped_count += 1

        debug_log(f"原始 memories 数量：{len(memories)}")
        debug_log(f"有效 memories 数量：{len(normalized_memories)}")
        debug_log(f"跳过非法 memories 数量：{skipped_count}")
        debug_log("长期记忆加载完成")

        return cls(path=memory_path, memories=normalized_memories)

    def save(self) -> None:
        debug_log("准备保存长期记忆")
        debug_log(f"保存路径：{self.path}")
        debug_log(f"当前长期记忆数量：{len(self.memories)}")

        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "memories": self.memories,
        }

        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        debug_log("长期记忆保存完成")

    def add_memory(
        self,
        memory_type: str,
        content: str,
        importance: float = 0.7,
        tags: list[str] | None = None,
        source: str = "manual",
    ) -> bool:
        debug_log("准备 add_memory")
        debug_log(f"memory_type：{memory_type}")
        debug_log(f"importance：{importance}")
        debug_log(f"source：{source}")
        debug_log(f"原始 content 长度：{len(content)}")
        debug_log(f"原始 content 预览：{preview_text(content)!r}")

        clean_content = content.strip()
        if not clean_content:
            debug_log("content 为空，跳过 add_memory")
            return False

        clean_tags = tags or []
        debug_log(f"tags：{clean_tags}")

        for index, memory in enumerate(self.memories):
            if memory.get("content") == clean_content:
                debug_log(f"发现重复 content，更新已有记忆，index={index}")
                memory["type"] = memory_type
                memory["importance"] = importance
                memory["tags"] = clean_tags
                memory["source"] = source
                memory["updated_at"] = _now_iso()
                self.save()
                debug_log("重复 content 已更新，但 add_memory 返回 False")
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

        debug_log("新增长期记忆成功")
        debug_log(f"新增后长期记忆数量：{len(self.memories)}")

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
        debug_log("准备 upsert_memory")
        debug_log(f"key：{key!r}")
        debug_log(f"memory_type：{memory_type}")
        debug_log(f"importance：{importance}")
        debug_log(f"source：{source}")
        debug_log(f"原始 content 长度：{len(content)}")
        debug_log(f"原始 content 预览：{preview_text(content)!r}")

        clean_key = key.strip()
        clean_content = content.strip()

        if not clean_key:
            debug_log("key 为空，跳过 upsert_memory")
            return "skipped"

        if not clean_content:
            debug_log("content 为空，跳过 upsert_memory")
            return "skipped"

        clean_tags = tags or []
        debug_log(f"tags：{clean_tags}")

        for index, memory in enumerate(self.memories):
            if memory.get("key") == clean_key:
                debug_log(f"命中已有 key，执行更新，index={index}")
                debug_log(f"旧 content 预览：{preview_text(str(memory.get('content', '')))!r}")

                memory["type"] = memory_type
                memory["content"] = clean_content
                memory["importance"] = importance
                memory["tags"] = clean_tags
                memory["source"] = source
                memory["updated_at"] = _now_iso()

                self.save()

                debug_log("已有 key 更新完成，返回 updated")
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

        debug_log("未命中已有 key，新增长期记忆")
        debug_log(f"新增后长期记忆数量：{len(self.memories)}")

        self.save()
        return "added"

    def format_for_prompt(self, limit: int = 20) -> str:
        debug_log("准备 format_for_prompt")
        debug_log(f"limit：{limit}")
        debug_log(f"当前长期记忆总数：{len(self.memories)}")

        if not self.memories:
            debug_log("长期记忆为空，返回默认文本")
            return "暂无长期记忆。"

        sorted_memories = sorted(
            self.memories,
            key=lambda item: (
                float(item.get("importance", 0.0)),
                str(item.get("updated_at", "")),
            ),
            reverse=True,
        )

        debug_log("长期记忆已按 importance 和 updated_at 排序")
        debug_log(f"本次最多格式化数量：{min(limit, len(sorted_memories))}")

        lines: list[str] = []
        skipped_empty_content = 0

        for index, memory in enumerate(sorted_memories[:limit], start=1):
            memory_type = str(memory.get("type", "fact"))
            content = str(memory.get("content", "")).strip()
            importance = memory.get("importance", 0.0)
            key = str(memory.get("key", "")).strip()
            tags = memory.get("tags", [])

            if not content:
                skipped_empty_content += 1
                debug_log(f"第 {index} 条记忆 content 为空，跳过")
                continue

            tag_text = ""
            if isinstance(tags, list) and tags:
                tag_text = " | tags: " + ", ".join(str(tag) for tag in tags)

            key_text = f" | key: {key}" if key else ""

            debug_log(
                f"格式化第 {index} 条记忆："
                f"type={memory_type}, "
                f"importance={importance}, "
                f"key={key!r}, "
                f"content_preview={preview_text(content)!r}"
            )

            lines.append(
                f"{index}. [{memory_type}] {content} "
                f"(importance: {importance}{key_text}{tag_text})"
            )

        debug_log(f"format_for_prompt 完成，输出行数：{len(lines)}")
        debug_log(f"空 content 跳过数量：{skipped_empty_content}")

        return "\n".join(lines) if lines else "暂无长期记忆。"