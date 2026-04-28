# DeepSeek Agent

一个基于 DeepSeek API 的本地实验性 Agent 助手，具备短期上下文管理和 AI 驱动的长期记忆功能。

## 项目概述

本项目实现了一个运行在本地 Python 程序中的对话式 Agent。它通过 DeepSeek API 调用远程模型，并自行管理对话上下文与长期记忆。核心思路是用**双模型策略**：主推理任务交给 DeepSeek Pro（强模型），记忆提取任务交给 DeepSeek Flash（轻量模型），在保证回答质量的同时控制成本。

## 环境要求

- Python 3.10+
- 一个有效的 [DeepSeek API Key](https://platform.deepseek.com/api_keys)

依赖只有两个第三方包：

| 包 | 用途 |
|---|---|
| `openai` | 以 OpenAI 兼容模式调用 DeepSeek API |
| `python-dotenv` | 从 `.env` 文件加载 API Key |

## 快速开始（从零到跑起来）

```bash
# 1. 进入项目目录
cd agent

# 2. 创建虚拟环境（可选但推荐）
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置 API Key —— 在项目根目录创建 .env 文件，写入：
#    DEEPSEEK_API_KEY=sk-你的key

# 6. 启动
python chat.py
```

## 启动后你会看到

```
DeepSeek Agent 已启动。
主推理模型：DeepSeek Pro
记忆更新模型：DeepSeek Flash
输入 /exit 退出，/clear 清空短期上下文，/memory 查看长期记忆。
------------------------------------------------------------

你：
```

此时输入任何问题即可开始对话。每次回答后，Flash 模型会在后台判断本轮对话是否包含值得长期记住的信息，如果有则自动写入 `data/memory.json`。

## 内置命令

| 命令 | 功能 | 什么时候用 |
|------|------|-----------|
| `/exit` | 保存会话和记忆后退出 | 正常退出，会话可恢复 |
| `/clear` | 清空当前对话历史 | 话题切换，想从头开始 |
| `/memory` | 打印当前长期记忆 | 查看 Agent 记住了什么 |

## 验证记忆功能是否工作

你可以用以下对话测试记忆链路：

```
你：我叫张三，我目前在用 Python 开发一个数据分析项目。

助手：你好张三，了解了你的数据分析项目……

[记忆更新] 已新增 2 条长期记忆。

你：/memory

当前长期记忆：
- [fact] 用户名叫张三 (importance=0.85)
- [project] 用户正在用 Python 开发数据分析项目 (importance=0.8)
```

退出后重新启动，输入 `/memory`，记忆仍在——因为存到了 `data/memory.json` 文件里。

## 运行时生成的文件

```
agent/
├── data/
│   └── memory.json       # 长期记忆，重启不丢失
├── sessions/
│   └── default.json      # 上次会话的对话历史，重启后恢复上下文
```

- 删除 `sessions/default.json` → 清空对话历史（短期记忆）
- 删除 `data/memory.json` → 清空所有长期记忆
- 两个文件都在 `.gitignore` 中，不会被提交

## 项目架构

```
agent/
├── chat.py              # 主入口，交互式对话循环
├── context_manager.py   # 短期上下文管理器（会话历史）
├── llm_client.py        # DeepSeek API 客户端封装
├── memory_store.py      # 长期记忆存储（JSON 文件）
├── memory_writer.py     # AI 驱动的记忆提取组件
├── data/
│   └── memory.json      # 长期记忆持久化文件
├── sessions/
│   └── default.json     # 会话历史持久化文件
└── testcode/
    ├── test_memory_store.py   # 记忆存储单元测试
    └── test_memory_writer.py  # 记忆存储单元测试
```

### 核心模块

#### 1. `chat.py` — 主程序入口

交互式对话循环，负责串联所有组件：
- 启动时加载历史会话和长期记忆
- 每轮对话前将长期记忆注入 system prompt
- 调用 Pro 模型生成回答
- 每轮对话后调用 Flash 模型提取长期记忆

可调参数（在 `main()` 中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_chars` | 12000 | 上下文字符数上限，超出则截断旧消息 |
| `keep_last_messages` | 12 | 最少保留的最近消息条数 |
| `temperature` (Pro) | 0.3 | 推理温度，越低越确定 |
| `temperature` (Flash) | 0.0 | 记忆提取温度，0 保证输出稳定 |

#### 2. `context_manager.py` — 短期上下文管理

维护会话中的消息列表（system / user / assistant），功能包括：
- **截断策略**：优先保留最近 N 条消息，超出字符数上限则从旧到新删除，system prompt 始终保留
- **持久化**：会话保存到 `sessions/default.json`，下次启动自动恢复

#### 3. `memory_store.py` — 长期记忆存储

基于 JSON 文件的记忆存储，提供两类写入方式：
- `add_memory()` — 追加式写入，按内容去重（相同内容只保留一份，取较高 importance）
- `upsert_memory()` — 基于 key 的写入，同一 key 再次写入会覆盖（适合"项目状态"这类需要更新的记忆）
- 检索时按 importance + updated_at 降序排列，取前 N 条注入 prompt

#### 4. `memory_writer.py` — AI 驱动的记忆提取

每轮对话结束后，将本轮用户输入和助手回答发送给 Flash 模型，由模型输出 JSON 判断是否写入长期记忆。写入阈值：importance ≥ 0.65。

它会保存的信息类型：长期偏好、项目状态、明确要求记住的内容、稳定事实、学习计划。
它会自动过滤：寒暄、玩笑、一次性临时内容、普通问答。

#### 5. `llm_client.py` — API 客户端

- 从 `.env` 加载 `DEEPSEEK_API_KEY`
- 使用 OpenAI SDK 兼容模式连接 `https://api.deepseek.com`
- 定义两个模型常量：`deepseek-v4-pro`（主推理，temperature 0.3）和 `deepseek-v4-flash`（记忆提取，temperature 0.0）

## 数据流

```
用户输入
  │
  ▼
┌─────────────────────────────────────┐
│  chat.py 主循环                      │
│                                      │
│  1. context.add_user(输入)           │
│  2. build_reasoning_messages()       │
│     ├── 读取 ContextManager 历史     │
│     └── 注入 MemoryStore 长期记忆     │
│  3. chat_completion(Pro模型) → 回答  │
│  4. context.add_assistant(回答)      │
│  5. update_memory_with_flash()       │
│     ├── 将本轮对话发给 Flash 模型     │
│     └── 提取的记忆写入 MemoryStore    │
└─────────────────────────────────────┘
  │
  ▼
输出回答 + 更新长期记忆
```

## 常见问题

**Q: 启动报 `未找到 DEEPSEEK_API_KEY`？**
确保 `.env` 文件在项目根目录（和 `chat.py` 同级），内容格式为 `DEEPSEEK_API_KEY=sk-xxx`，等号两边不要加空格或引号。

**Q: 调用 Pro 模型失败？**
检查 API Key 是否有效、网络是否能访问 `api.deepseek.com`、账户余额是否充足。

**Q: 记忆更新一直显示"Flash 返回内容无法解析"？**
Flash 模型偶发输出格式不稳定，不影响主对话。可重试或降低 temperature（当前已设为 0.0）。

**Q: 想用其他模型？**
修改 `llm_client.py` 中的 `PRO_MODEL` 和 `FLASH_MODEL` 常量，以及 `create_client()` 中的 `base_url` 即可切换至其他 OpenAI 兼容接口。
