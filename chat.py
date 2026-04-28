from pathlib import Path

from context_manager import ContextManager
from llm_client import PRO_MODEL, chat_completion, create_client
from memory_store import MemoryStore
from memory_writer import update_memory_with_flash


BASE_DIR = Path(__file__).parent
SESSION_PATH = BASE_DIR / "sessions" / "default.json"
MEMORY_PATH = BASE_DIR / "data" / "memory.json"


SYSTEM_PROMPT = """
你是一个通过 DeepSeek API 调用的本地实验性 Agent 助手。
你运行在用户本地 Python 程序中，但模型能力来自远程 DeepSeek API。

要求：
1. 回答要清晰、直接。
2. 如果用户在测试上下文记忆，你要基于历史对话回答。
3. 不要假装知道没有出现在上下文里的信息。
4. 当用户问你是什么模型时，说明你是通过 DeepSeek API 调用的模型助手，不要声称自己是本地部署模型。
""".strip()


def build_reasoning_messages(
    context: ContextManager,
    memory_store: MemoryStore,
) -> list[dict[str, str]]:
    long_term_memory = memory_store.format_for_prompt(limit=20)

    memory_message = {
        "role": "system",
        "content": (
            "以下是用户的长期记忆，回答时如果相关可以参考；"
            "如果不相关，不要强行使用。\n\n"
            f"{long_term_memory}"
        ),
    }

    base_messages = context.build_messages()

    # 保留原 system，再插入长期记忆
    return [base_messages[0], memory_message] + base_messages[1:]



def main() -> None:
    client = create_client()
    memory_store = MemoryStore.load(MEMORY_PATH)

    context = ContextManager.load(
        path=SESSION_PATH,
        system_prompt=SYSTEM_PROMPT,
        max_chars=12000,
        keep_last_messages=12,
    )

    print("DeepSeek Agent 已启动。")
    print("主推理模型：DeepSeek Pro")
    print("记忆更新模型：DeepSeek Flash")
    print("输入 /exit 退出，/clear 清空短期上下文，/memory 查看长期记忆。")
    print("-" * 60)

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            context.save(SESSION_PATH)
            memory_store.save()
            print("已保存会话和长期记忆，退出。")
            break

        if user_input == "/clear":
            context.clear()
            context.save(SESSION_PATH)
            print("短期上下文已清空。")
            continue

        if user_input == "/memory":
            print("\n当前长期记忆：")
            print(memory_store.format_for_prompt(limit=50))
            continue

        context.add_user(user_input)

        try:
            reasoning_messages = build_reasoning_messages(context, memory_store)
            answer = chat_completion(
                client=client,
                model=PRO_MODEL,
                messages=reasoning_messages,
                temperature=0.3,
            )
        except Exception as exc:
            print(f"调用 Pro 模型失败：{exc}")
            continue

        context.add_assistant(answer)
        context.save(SESSION_PATH)

        print(f"\n助手：{answer}")

        try:
            update_memory_with_flash(
                client=client,
                memory_store=memory_store,
                user_input=user_input,
                assistant_answer=answer,
            )
        except Exception as exc:
            print(f"\n[记忆更新] 调用 Flash 失败：{exc}")


if __name__ == "__main__":
    main()