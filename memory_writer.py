from __future__ import annotations

import json
import re
from typing import Any

from llm_client import FLASH_MODEL, chat_completion
from memory_store import MemoryStore

MEMORY_UPDATE_PROMPT = """
你是一个本地 Agent 的记忆更新器。
你的任务是判断本轮对话中是否有值得写入长期记忆的信息。

只保存长期有价值的信息，例如：
1. 用户明确表达的长期偏好
2. 用户正在做的项目状态
3. 用户明确要求记住的信息
4. 对未来回答有帮助的稳定事实
5. 用户的学习目标、技术路线、长期计划

不要保存：
1. 寒暄
2. 玩笑
3. 一次性临时内容
4. 没有长期价值的普通问答
5. 敏感隐私信息，除非用户明确要求保存
6. 助手提出的建议，除非用户明确表示采纳或决定采用

你必须只输出 JSON，不要输出解释文字。

重要规则：
- 如果本轮信息是在更新已有事实，请使用相同的 key。
- key 必须稳定、简短、英文小写，用下划线连接。
- 同一个事实的不同表达必须使用同一个 key。
- 如果没有稳定 key，可以让 key 为空字符串。

JSON 格式如下：
{
  "should_update_memory": true,
  "memories": [
    {
      "action": "add_or_update",
      "key": "stable_memory_key",
      "type": "preference | project | fact | learning | technical_note",
      "content": "要写入长期记忆的内容",
      "importance": 0.8,
      "tags": ["tag1", "tag2"]
    }
  ]
}

如果没有值得保存的内容，输出：
{
  "should_update_memory": false,
  "memories": []
}
""".strip()


MIN_IMPORTANCE = 0.65
MEMORY_SOURCE = "flash_memory_writer"


def extract_json_block(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    clean_text = text.strip()

    try:
        data = json.loads(clean_text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        clean_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        try:
            data = json.loads(fenced_match.group(1))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(clean_text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def _safe_float(value: Any, default: float = 0.7) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default

    if number < 0:
        return 0.0

    if number > 1:
        return 1.0

    return number


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value).strip()


def _normalize_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []

    result: list[str] = []
    for tag in tags:
        tag_text = _safe_str(tag)
        if tag_text:
            result.append(tag_text)

    return result


def _write_memory_item(
    memory_store: MemoryStore,
    item: dict[str, Any],
    min_importance: float = MIN_IMPORTANCE,
) -> str:
    if not isinstance(item, dict):
        return "skipped"

    action = _safe_str(item.get("action", "add_or_update")).lower()
    if action not in {"", "add", "update", "add_or_update"}:
        return "skipped"

    content = _safe_str(item.get("content"))
    if not content:
        return "skipped"

    importance = _safe_float(item.get("importance", 0.7), default=0.7)
    if importance < min_importance:
        return "skipped"

    key = _safe_str(item.get("key"))
    memory_type = _safe_str(item.get("type", "fact"), default="fact") or "fact"
    tags = _normalize_tags(item.get("tags", []))

    if key:
        return memory_store.upsert_memory(
            key=key,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags,
            source=MEMORY_SOURCE,
        )

    added = memory_store.add_memory(
        memory_type=memory_type,
        content=content,
        importance=importance,
        tags=tags,
        source=MEMORY_SOURCE,
    )

    return "added" if added else "duplicated"


def update_memory_with_flash(
    client: Any,
    memory_store: MemoryStore,
    user_input: str,
    assistant_answer: str,
) -> None:
    messages = [
        {"role": "system", "content": MEMORY_UPDATE_PROMPT},
        {
            "role": "user",
            "content": (
                "请判断以下本轮对话是否需要更新长期记忆。\n\n"
                f"用户输入：\n{user_input}\n\n"
                f"助手回答：\n{assistant_answer}"
            ),
        },
    ]

    raw = chat_completion(
        client=client,
        model=FLASH_MODEL,
        messages=messages,
        temperature=0.0,
    )

    data = extract_json_block(raw)
    if not data:
        print("\n[记忆更新] Flash 返回内容无法解析，已跳过。")
        return

    if not data.get("should_update_memory"):
        print("\n[记忆更新] 本轮无长期记忆更新。")
        return

    memories = data.get("memories", [])
    if not isinstance(memories, list):
        print("\n[记忆更新] memories 字段格式错误，已跳过。")
        return

    stats = {
        "added": 0,
        "updated": 0,
        "duplicated": 0,
        "skipped": 0,
    }

    for item in memories:
        result = _write_memory_item(memory_store, item)
        if result in stats:
            stats[result] += 1
        else:
            stats["skipped"] += 1

    if stats["added"] or stats["updated"]:
        print(
            "\n[记忆更新] 完成："
            f"新增 {stats['added']} 条，"
            f"更新 {stats['updated']} 条，"
            f"重复 {stats['duplicated']} 条，"
            f"跳过 {stats['skipped']} 条。"
        )
    else:
        print(
            "\n[记忆更新] 没有写入新的长期记忆："
            f"重复 {stats['duplicated']} 条，"
            f"跳过 {stats['skipped']} 条。"
        )
