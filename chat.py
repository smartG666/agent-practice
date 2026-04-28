from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from context_manager import ContextManager
from llm_client import PRO_MODEL, chat_completion, create_client
from memory_store import MemoryStore
from memory_writer import update_memory_with_flash

BASE_DIR = Path(__file__).parent
SESSION_PATH = BASE_DIR / "sessions" / "default.json"
MEMORY_PATH = BASE_DIR / "data" / "memory.json"
DEBUG_LOG_DIR = BASE_DIR / "debug_logs"

# 是否开启调试日志
# 不想看日志时，改成 False 即可
DEBUG = True

# 是否把 Pro 模型完整输入打印到终端
PRINT_FULL_PRO_INPUT = True

# 是否把 Pro 模型完整输入保存到文件
SAVE_FULL_PRO_INPUT = True


SYSTEM_PROMPT = """
你是一个通过 DeepSeek API 调用的本地实验性 Agent 助手。
你运行在用户本地 Python 程序中，但模型能力来自远程 DeepSeek API。

要求：
1. 回答要清晰、直接。
2. 如果用户在测试上下文记忆，你要基于历史对话回答。
3. 不要假装知道没有出现在上下文里的信息。
4. 当用户问你是什么模型时，说明你是通过 DeepSeek API 调用的模型助手，不要声称自己是本地部署模型。
""".strip()


def debug_log(message: str) -> None:
    """
    打印 chat.py 调试日志。
    """
    if DEBUG:
        print(f"[DEBUG][chat] {message}")


def preview_text(text: str, limit: int = 80) -> str:
    """
    生成文本预览，避免日志太长。
    """
    return text.replace("\n", " ")[:limit]


def safe_memory_count(memory_store: MemoryStore) -> str:
    """
    尝试获取长期记忆数量。

    这里用 getattr 是为了避免 MemoryStore 内部字段名变化导致程序崩溃。
    """
    memories = getattr(memory_store, "memories", None)

    if isinstance(memories, list):
        return str(len(memories))

    return "未知"


def debug_messages_summary(
    messages: list[dict[str, str]],
    label: str,
) -> None:
    """
    打印 messages 的概要，不打印完整内容，避免日志太长。
    """
    if not DEBUG:
        return

    print(f"[DEBUG][chat] {label} messages 概要：")
    print(f"[DEBUG][chat] messages 总数：{len(messages)}")

    for index, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        content_length = len(content)
        preview = preview_text(content)

        print(
            f"[DEBUG][chat]   #{index} "
            f"role={role}, "
            f"content_length={content_length}, "
            f"preview={preview!r}"
        )


def dump_pro_messages(messages: list[dict[str, str]]) -> None:
    """
    打印并保存即将发送给 Pro 模型的完整 messages。

    注意：
    这里的 messages 就是 Pro 模型实际收到的输入。
    它不是一个拼接后的大字符串，而是 Chat Completions 格式的消息列表。
    """
    if not DEBUG:
        return

    DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(
        messages,
        ensure_ascii=False,
        indent=2,
    )

    latest_path = DEBUG_LOG_DIR / "pro_input_latest.json"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = DEBUG_LOG_DIR / f"pro_input_{timestamp}.json"

    if SAVE_FULL_PRO_INPUT:
        latest_path.write_text(json_text, encoding="utf-8")
        history_path.write_text(json_text, encoding="utf-8")

        print("\n" + "=" * 80)
        print("[DEBUG][Pro 输入] 即将发送给 Pro 模型的完整 messages 已保存")
        print(f"[DEBUG][Pro 输入] 最新文件：{latest_path}")
        print(f"[DEBUG][Pro 输入] 历史文件：{history_path}")
        print("=" * 80)

    if PRINT_FULL_PRO_INPUT:
        print("\n" + "=" * 80)
        print("[DEBUG][Pro 输入] 即将发送给 Pro 模型的完整 messages")
        print("=" * 80)
        print(json_text)
        print("=" * 80)
        print("[DEBUG][Pro 输入] messages 打印结束")
        print("=" * 80 + "\n")


def build_reasoning_messages(
    context: ContextManager,
    memory_store: MemoryStore,
) -> list[dict[str, str]]:
    """
    构造发送给主推理模型的 messages。

    结构：
    1. 原始 system prompt
    2. 长期记忆 system prompt
    3. 最近短期上下文
    """
    debug_log("开始构造主推理模型 messages")

    long_term_memory = memory_store.format_for_prompt(limit=20)
    debug_log(f"长期记忆 prompt 构造完成，长度：{len(long_term_memory)}")

    memory_message: dict[str, str] = {
        "role": "system",
        "content": (
            "以下是用户的长期记忆，回答时如果相关可以参考；"
            "如果不相关，不要强行使用。\n\n"
            f"{long_term_memory}"
        ),
    }

    base_messages = context.build_messages()
    debug_log(f"短期上下文 messages 数量：{len(base_messages)}")

    if not base_messages:
        debug_log("短期上下文为空，仅返回长期记忆 message")
        return [memory_message]

    result: list[dict[str, str]] = []

    first_message = base_messages[0]
    result.append(
        {
            "role": first_message["role"],
            "content": first_message["content"],
        }
    )

    result.append(memory_message)

    for message in base_messages[1:]:
        result.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    debug_log(f"主推理模型 messages 构造完成，总数量：{len(result)}")

    return result


def main() -> None:
    debug_log("程序启动")
    debug_log(f"项目目录：{BASE_DIR}")
    debug_log(f"短期上下文文件：{SESSION_PATH}")
    debug_log(f"长期记忆文件：{MEMORY_PATH}")
    debug_log(f"调试日志目录：{DEBUG_LOG_DIR}")

    debug_log("开始创建 DeepSeek API 客户端")
    client: Any = create_client()
    debug_log("DeepSeek API 客户端创建完成")

    debug_log("开始加载长期记忆")
    memory_store = MemoryStore.load(MEMORY_PATH)
    debug_log(f"长期记忆加载完成，当前数量：{safe_memory_count(memory_store)}")

    debug_log("开始加载短期上下文")
    context = ContextManager.load(
        path=SESSION_PATH,
        system_prompt=SYSTEM_PROMPT,
        max_chars=12000,
        keep_last_messages=12,
    )
    debug_log("短期上下文加载完成")

    print("DeepSeek Agent 已启动。")
    print("主推理模型：DeepSeek Pro")
    print("记忆更新模型：DeepSeek Flash")
    print("输入 /exit 退出，/clear 清空短期上下文，/memory 查看长期记忆。")
    print("-" * 60)

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            debug_log("收到空输入，跳过本轮")
            continue

        debug_log(f"收到用户输入，长度：{len(user_input)}")
        debug_log(f"用户输入预览：{user_input[:80]!r}")

        if user_input == "/exit":
            debug_log("收到 /exit 命令，准备保存短期上下文和长期记忆")
            context.save(SESSION_PATH)
            memory_store.save()
            debug_log("短期上下文和长期记忆保存完成")
            print("已保存会话和长期记忆，退出。")
            break

        if user_input == "/clear":
            debug_log("收到 /clear 命令，准备清空短期上下文")
            context.clear()
            context.save(SESSION_PATH)
            debug_log("短期上下文已清空并保存")
            print("短期上下文已清空。")
            continue

        if user_input == "/memory":
            debug_log("收到 /memory 命令，准备查看长期记忆")
            debug_log(f"当前长期记忆数量：{safe_memory_count(memory_store)}")
            print("\n当前长期记忆：")
            print(memory_store.format_for_prompt(limit=50))
            continue

        debug_log("1. 准备把用户输入加入短期上下文")
        context.add_user(user_input)
        debug_log("2. 用户输入已加入短期上下文")

        try:
            debug_log("3. 准备构造 Pro 主推理 messages")
            reasoning_messages = build_reasoning_messages(
                context=context,
                memory_store=memory_store,
            )

            debug_messages_summary(
                messages=reasoning_messages,
                label="Pro 主推理",
            )

            # 关键点：
            # 这里打印 / 保存的 reasoning_messages
            # 就是即将发送给 Pro 模型的完整输入。
            dump_pro_messages(reasoning_messages)

            debug_log(f"4. 准备调用 Pro 模型：{PRO_MODEL}")
            answer = chat_completion(
                client=client,
                model=PRO_MODEL,
                messages=reasoning_messages,
                temperature=0.3,
            )
            debug_log("5. Pro 模型调用完成")
            debug_log(f"Pro 回答长度：{len(answer)}")

        except Exception as exc:
            debug_log(f"Pro 模型调用异常：{exc}")
            print(f"调用 Pro 模型失败：{exc}")
            continue

        debug_log("6. 准备把助手回答加入短期上下文")
        context.add_assistant(answer)
        debug_log("7. 助手回答已加入短期上下文")

        debug_log("8. 准备保存短期上下文")
        context.save(SESSION_PATH)
        debug_log("9. 短期上下文保存完成")

        print(f"\n助手：{answer}")

        try:
            before_memory_count = safe_memory_count(memory_store)
            debug_log(f"10. 准备调用 Flash 更新长期记忆，更新前数量：{before_memory_count}")

            update_memory_with_flash(
                client=client,
                memory_store=memory_store,
                user_input=user_input,
                assistant_answer=answer,
            )

            after_memory_count = safe_memory_count(memory_store)
            debug_log(f"11. Flash 长期记忆更新流程结束，更新后数量：{after_memory_count}")

        except Exception as exc:
            debug_log(f"Flash 记忆更新异常：{exc}")
            print(f"\n[记忆更新] 调用 Flash 失败：{exc}")


if __name__ == "__main__":
    main()
