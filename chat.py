import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from context_manager import ContextManager


BASE_DIR = Path(__file__).parent
SESSION_PATH = BASE_DIR / "sessions" / "default.json"


SYSTEM_PROMPT = """
你是一个本地 Agent 学习助手。
要求：
1. 回答要清晰、直接。
2. 如果用户在测试上下文记忆，你要基于历史对话回答。
3. 不要假装知道没有出现在上下文里的信息。
""".strip()


def create_client() -> OpenAI:
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(f"未找到 DEEPSEEK_API_KEY，请检查：{env_path}")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def ask_model(client: OpenAI, context: ContextManager) -> str:
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=context.build_messages(),
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if content is None:
        return ""

    return content


def main() -> None:
    client = create_client()

    context = ContextManager.load(
        path=SESSION_PATH,
        system_prompt=SYSTEM_PROMPT,
        max_chars=12000,
        keep_last_messages=12,
    )

    print("DeepSeek Chat 已启动。")
    print("输入 /exit 退出，/clear 清空上下文，/show 查看当前上下文条数。")
    print("-" * 60)

    while True:
        user_input = input("\n你：").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            context.save(SESSION_PATH)
            print("已保存会话，退出。")
            break

        if user_input == "/clear":
            context.clear()
            context.save(SESSION_PATH)
            print("上下文已清空。")
            continue

        if user_input == "/show":
            print(f"当前历史消息数：{len(context.messages)}")
            print(f"实际发送消息数：{len(context.build_messages())}")
            print(f"会话文件：{SESSION_PATH}")
            continue

        context.add_user(user_input)

        try:
            answer = ask_model(client, context)
        except Exception as exc:
            print(f"调用模型失败：{exc}")
            continue

        context.add_assistant(answer)
        context.save(SESSION_PATH)

        print(f"\n助手：{answer}")


if __name__ == "__main__":
    main()