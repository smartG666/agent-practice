from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import memory_writer
from memory_store import MemoryStore


class TestExtractJsonBlock(unittest.TestCase):
    def test_extract_pure_json(self) -> None:
        raw = '{"should_update_memory": false, "memories": []}'
        data = memory_writer.extract_json_block(raw)

        assert data is not None
        self.assertIsInstance(data, dict)
        self.assertFalse(data["should_update_memory"])
        self.assertEqual(data["memories"], [])

    def test_extract_fenced_json(self) -> None:
        raw = """```json
{
  "should_update_memory": true,
  "memories": []
}
```"""
        data = memory_writer.extract_json_block(raw)

        assert data is not None
        self.assertIsInstance(data, dict)
        self.assertTrue(data["should_update_memory"])

    def test_extract_json_with_extra_text(self) -> None:
        raw = """这是模型误输出的解释文字：
{
  "should_update_memory": false,
  "memories": []
}
结束。"""
        data = memory_writer.extract_json_block(raw)

        assert data is not None
        self.assertIsInstance(data, dict)
        self.assertFalse(data["should_update_memory"])

    def test_invalid_json_returns_none(self) -> None:
        raw = "not json"
        data = memory_writer.extract_json_block(raw)

        self.assertIsNone(data)


class TestUpdateMemoryWithFlash(unittest.TestCase):
    def _new_store(self, tmpdir: str) -> MemoryStore:
        path = Path(tmpdir) / "memory.json"
        return MemoryStore(path=path, memories=[])

    def test_update_memory_uses_key_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._new_store(tmpdir)

            first_response = json.dumps(
                {
                    "should_update_memory": True,
                    "memories": [
                        {
                            "action": "add_or_update",
                            "key": "agent_project_current_stage",
                            "type": "project",
                            "content": "用户正在开发本地 Agent 项目，当前阶段是 JSON 记忆。",
                            "importance": 0.8,
                            "tags": ["agent", "memory"],
                        }
                    ],
                },
                ensure_ascii=False,
            )

            second_response = json.dumps(
                {
                    "should_update_memory": True,
                    "memories": [
                        {
                            "action": "add_or_update",
                            "key": "agent_project_current_stage",
                            "type": "project",
                            "content": "用户正在开发本地 Agent 项目，当前阶段是完善 memory_writer。",
                            "importance": 0.9,
                            "tags": ["agent", "memory_writer"],
                        }
                    ],
                },
                ensure_ascii=False,
            )

            with patch.object(memory_writer, "chat_completion", return_value=first_response):
                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="我在做本地 Agent 项目",
                    assistant_answer="好的",
                )

            self.assertEqual(len(store.memories), 1)
            self.assertEqual(store.memories[0]["key"], "agent_project_current_stage")
            self.assertIn("JSON 记忆", store.memories[0]["content"])

            with patch.object(memory_writer, "chat_completion", return_value=second_response):
                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="现在我要完善 memory_writer",
                    assistant_answer="好的",
                )

            self.assertEqual(len(store.memories), 1)
            self.assertEqual(store.memories[0]["key"], "agent_project_current_stage")
            self.assertIn("完善 memory_writer", store.memories[0]["content"])
            self.assertEqual(store.memories[0]["importance"], 0.9)
            self.assertEqual(store.memories[0]["tags"], ["agent", "memory_writer"])

    def test_no_key_fallback_to_add_memory_and_deduplicate_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._new_store(tmpdir)

            response = json.dumps(
                {
                    "should_update_memory": True,
                    "memories": [
                        {
                            "action": "add_or_update",
                            "key": "",
                            "type": "preference",
                            "content": "用户喜欢用中文解释复杂技术问题。",
                            "importance": 0.8,
                            "tags": ["style"],
                        }
                    ],
                },
                ensure_ascii=False,
            )

            with patch.object(memory_writer, "chat_completion", return_value=response):
                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="复杂问题用中文",
                    assistant_answer="好的",
                )

                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="复杂问题用中文",
                    assistant_answer="好的",
                )

            self.assertEqual(len(store.memories), 1)
            self.assertNotIn("key", store.memories[0])
            self.assertIn("中文解释", store.memories[0]["content"])

    def test_low_importance_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._new_store(tmpdir)

            response = json.dumps(
                {
                    "should_update_memory": True,
                    "memories": [
                        {
                            "action": "add_or_update",
                            "key": "temporary_chat",
                            "type": "fact",
                            "content": "用户刚才说了一句临时闲聊。",
                            "importance": 0.3,
                            "tags": ["temp"],
                        }
                    ],
                },
                ensure_ascii=False,
            )

            with patch.object(memory_writer, "chat_completion", return_value=response):
                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="随便聊聊",
                    assistant_answer="好的",
                )

            self.assertEqual(store.memories, [])

    def test_invalid_model_json_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._new_store(tmpdir)

            with patch.object(memory_writer, "chat_completion", return_value="not json"):
                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="hello",
                    assistant_answer="world",
                )

            self.assertEqual(store.memories, [])

    def test_invalid_importance_uses_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._new_store(tmpdir)

            response = json.dumps(
                {
                    "should_update_memory": True,
                    "memories": [
                        {
                            "action": "add_or_update",
                            "key": "user_learning_goal",
                            "type": "learning",
                            "content": "用户正在学习 Python 输入输出和矩阵处理。",
                            "importance": "high",
                            "tags": ["python"],
                        }
                    ],
                },
                ensure_ascii=False,
            )

            with patch.object(memory_writer, "chat_completion", return_value=response):
                memory_writer.update_memory_with_flash(
                    client=None,
                    memory_store=store,
                    user_input="我在学 Python",
                    assistant_answer="好的",
                )

            self.assertEqual(len(store.memories), 1)
            self.assertEqual(store.memories[0]["importance"], 0.7)


if __name__ == "__main__":
    unittest.main()
