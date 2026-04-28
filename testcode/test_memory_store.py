from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory_store import MemoryStore


class TestMemoryStore(unittest.TestCase):
    def test_load_creates_empty_store_when_file_not_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = MemoryStore.load(path)

            self.assertEqual(store.path, path)
            self.assertEqual(store.memories, [])

    def test_upsert_adds_then_updates_same_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = MemoryStore(path=path, memories=[])

            result1 = store.upsert_memory(
                key="agent_project_current_stage",
                memory_type="project",
                content="第一版内容",
                importance=0.7,
                tags=["agent"],
                source="test",
            )

            result2 = store.upsert_memory(
                key="agent_project_current_stage",
                memory_type="project",
                content="第二版内容",
                importance=0.9,
                tags=["agent", "updated"],
                source="test",
            )

            self.assertEqual(result1, "added")
            self.assertEqual(result2, "updated")
            self.assertEqual(len(store.memories), 1)
            self.assertEqual(store.memories[0]["key"], "agent_project_current_stage")
            self.assertEqual(store.memories[0]["content"], "第二版内容")
            self.assertEqual(store.memories[0]["importance"], 0.9)
            self.assertEqual(store.memories[0]["tags"], ["agent", "updated"])

    def test_add_memory_deduplicates_same_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = MemoryStore(path=path, memories=[])

            added1 = store.add_memory(
                memory_type="preference",
                content="用户喜欢中文解释。",
                importance=0.7,
                tags=["style"],
                source="test",
            )

            added2 = store.add_memory(
                memory_type="preference",
                content="用户喜欢中文解释。",
                importance=0.9,
                tags=["style"],
                source="test",
            )

            self.assertTrue(added1)
            self.assertFalse(added2)
            self.assertEqual(len(store.memories), 1)
            self.assertEqual(store.memories[0]["importance"], 0.9)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            store = MemoryStore(path=path, memories=[])

            store.upsert_memory(
                key="user_python_learning_goal",
                memory_type="learning",
                content="用户正在学习 Python。",
                importance=0.8,
                tags=["python"],
                source="test",
            )

            loaded_store = MemoryStore.load(path)

            self.assertEqual(len(loaded_store.memories), 1)
            self.assertEqual(
                loaded_store.memories[0]["key"],
                "user_python_learning_goal",
            )


if __name__ == "__main__":
    unittest.main()
