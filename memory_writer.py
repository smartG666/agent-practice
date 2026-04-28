from __future__ import annotations

import json
import re
from typing import Any

from llm_client import FLASH_MODEL, chat_completion
from memory_store import MemoryStore


# 是否开启 memory_writer 调试日志
DEBUG = True


def debug_log(message: str) -> None:
    """
    打印 memory_writer 模块调试日志。
    """
    if DEBUG:
        print(f"[DEBUG][memory_writer] {message}")


def preview_text(text: str, limit: int = 120) -> str:
    """
    生成文本预览，避免日志太长。
    """
    return text.replace("\n", " ")[:limit]


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
    debug_log("开始 extract_json_block")

    if not text:
        debug_log("输入 text 为空，返回 None")
        return None

    clean_text = text.strip()

    debug_log(f"Flash 原始返回长度：{len(text)}")
    debug_log(f"Flash 原始返回预览：{preview_text(clean_text, limit=200)!r}")

    try:
        debug_log("尝试直接 json.loads 解析完整文本")
        data = json.loads(clean_text)
        if isinstance(data, dict):
            debug_log("完整文本 JSON 解析成功")
            return data

        debug_log("完整文本 JSON 解析成功，但结果不是 dict，返回 None")
        return None

    except json.JSONDecodeError as exc:
        debug_log(f"完整文本 JSON 解析失败：{exc}")

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        clean_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        debug_log("检测到 fenced JSON 代码块，尝试解析")
        try:
            data = json.loads(fenced_match.group(1))
            if isinstance(data, dict):
                debug_log("fenced JSON 解析成功")
                return data

            debug_log("fenced JSON 解析成功，但结果不是 dict")
            return None

        except json.JSONDecodeError as exc:
            debug_log(f"fenced JSON 解析失败：{exc}")

    start = clean_text.find("{")
    end = clean_text.rfind("}")

    if start != -1 and end != -1 and end > start:
        debug_log("尝试截取第一个 { 到最后一个 } 之间的内容解析")
        debug_log(f"JSON 候选片段起止位置：start={start}, end={end}")

        try:
            data = json.loads(clean_text[start : end + 1])
            if isinstance(data, dict):
                debug_log("截取 JSON 解析成功")
                return data

            debug_log("截取 JSON 解析成功，但结果不是 dict")
            return None

        except json.JSONDecodeError as exc:
            debug_log(f"截取 JSON 解析失败：{exc}")
            return None

    debug_log("没有找到可解析 JSON，返回 None")
    return None


def _safe_float(value: Any, default: float = 0.7) -> float:
    debug_log(f"_safe_float 输入：{value!r}, default={default}")

    try:
        number = float(value)
    except (TypeError, ValueError):
        debug_log("_safe_float 转换失败，使用默认值")
        number = default

    if number < 0:
        debug_log("_safe_float 小于 0，裁剪为 0.0")
        return 0.0

    if number > 1:
        debug_log("_safe_float 大于 1，裁剪为 1.0")
        return 1.0

    debug_log(f"_safe_float 输出：{number}")
    return number


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        debug_log(f"_safe_str 输入 None，返回 default={default!r}")
        return default

    result = str(value).strip()
    debug_log(f"_safe_str 输出：{result!r}")
    return result


def _normalize_tags(tags: Any) -> list[str]:
    debug_log(f"_normalize_tags 输入：{tags!r}")

    if not isinstance(tags, list):
        debug_log("_normalize_tags 输入不是 list，返回空列表")
        return []

    result: list[str] = []

    for tag in tags:
        tag_text = _safe_str(tag)
        if tag_text:
            result.append(tag_text)

    debug_log(f"_normalize_tags 输出：{result}")
    return result


def _write_memory_item(
    memory_store: MemoryStore,
    item: dict[str, Any],
    min_importance: float = MIN_IMPORTANCE,
) -> str:
    debug_log("开始 _write_memory_item")

    if not isinstance(item, dict):
        debug_log("item 不是 dict，跳过")
        return "skipped"

    debug_log(f"原始 item keys：{list(item.keys())}")

    action = _safe_str(item.get("action", "add_or_update")).lower()
    debug_log(f"action：{action!r}")

    if action not in {"", "add", "update", "add_or_update"}:
        debug_log("action 不在允许范围内，跳过")
        return "skipped"

    content = _safe_str(item.get("content"))
    debug_log(f"content 长度：{len(content)}")
    debug_log(f"content 预览：{preview_text(content)!r}")

    if not content:
        debug_log("content 为空，跳过")
        return "skipped"

    importance = _safe_float(item.get("importance", 0.7), default=0.7)
    debug_log(f"importance：{importance}, min_importance：{min_importance}")

    if importance < min_importance:
        debug_log("importance 低于阈值，跳过")
        return "skipped"

    key = _safe_str(item.get("key"))
    memory_type = _safe_str(item.get("type", "fact"), default="fact") or "fact"
    tags = _normalize_tags(item.get("tags", []))

    debug_log(f"key：{key!r}")
    debug_log(f"memory_type：{memory_type!r}")
    debug_log(f"tags：{tags}")

    if key:
        debug_log("存在 key，调用 memory_store.upsert_memory")
        result = memory_store.upsert_memory(
            key=key,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags,
            source=MEMORY_SOURCE,
        )
        debug_log(f"upsert_memory 返回：{result}")
        return result

    debug_log("key 为空，调用 memory_store.add_memory")

    added = memory_store.add_memory(
        memory_type=memory_type,
        content=content,
        importance=importance,
        tags=tags,
        source=MEMORY_SOURCE,
    )

    result = "added" if added else "duplicated"
    debug_log(f"add_memory 返回 added={added}，最终结果：{result}")

    return result


def update_memory_with_flash(
    client: Any,
    memory_store: MemoryStore,
    user_input: str,
    assistant_answer: str,
) -> None:
    debug_log("开始 update_memory_with_flash")
    debug_log(f"user_input 长度：{len(user_input)}")
    debug_log(f"user_input 预览：{preview_text(user_input)!r}")
    debug_log(f"assistant_answer 长度：{len(assistant_answer)}")
    debug_log(f"assistant_answer 预览：{preview_text(assistant_answer)!r}")
    debug_log(f"当前长期记忆数量：{len(memory_store.memories)}")

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

    debug_log("Flash 记忆更新 messages 构造完成")
    debug_log(f"messages 数量：{len(messages)}")

    for index, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        debug_log(
            f"message #{index}: "
            f"role={role}, "
            f"content_length={len(content)}, "
            f"preview={preview_text(content)!r}"
        )

    debug_log(f"准备调用 Flash 模型：{FLASH_MODEL}")

    raw = chat_completion(
        client=client,
        model=FLASH_MODEL,
        messages=messages,
        temperature=0.0,
    )

    debug_log("Flash 模型调用完成")
    debug_log(f"Flash 返回长度：{len(raw)}")
    debug_log(f"Flash 返回预览：{preview_text(raw, limit=200)!r}")

    data = extract_json_block(raw)

    if not data:
        debug_log("Flash 返回内容无法解析为 JSON，跳过长期记忆更新")
        print("\n[记忆更新] Flash 返回内容无法解析，已跳过。")
        return

    debug_log(f"解析后的 JSON keys：{list(data.keys())}")
    debug_log(f"should_update_memory：{data.get('should_update_memory')}")

    if not data.get("should_update_memory"):
        debug_log("should_update_memory 为 False，本轮无长期记忆更新")
        print("\n[记忆更新] 本轮无长期记忆更新。")
        return

    memories = data.get("memories", [])
    debug_log(f"memories 字段类型：{type(memories).__name__}")

    if not isinstance(memories, list):
        debug_log("memories 字段不是 list，跳过")
        print("\n[记忆更新] memories 字段格式错误，已跳过。")
        return

    debug_log(f"Flash 返回 memories 数量：{len(memories)}")

    stats = {
        "added": 0,
        "updated": 0,
        "duplicated": 0,
        "skipped": 0,
    }

    for index, item in enumerate(memories, start=1):
        debug_log(f"开始处理第 {index} 条 memory item")
        result = _write_memory_item(memory_store, item)
        debug_log(f"第 {index} 条 memory item 处理结果：{result}")

        if result in stats:
            stats[result] += 1
        else:
            stats["skipped"] += 1

    debug_log(f"记忆更新统计：{stats}")
    debug_log(f"更新后长期记忆数量：{len(memory_store.memories)}")

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