import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(f"未找到 DEEPSEEK_API_KEY，请检查 .env 文件路径：{env_path}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "你是一个简洁、可靠的编程助手。"},
            {"role": "user", "content": "用一句话解释什么是 API。"},
        ],
        temperature=0.3,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()