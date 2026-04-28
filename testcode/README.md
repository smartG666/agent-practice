# 测试代码说明

本目录包含对 `MemoryStore` 模块的单元测试。

## 测试内容

### `test_memory_store.py`

测试 `MemoryStore` 的核心功能——记忆的增删去重逻辑，包含两个测试用例：

#### 1. `test_upsert_adds_then_updates_same_key`

测试基于 key 的记忆写入（`upsert_memory`）：

- 第一次用某个 key 写入时，应返回 `"added"` 并创建新记忆
- 第二次用相同 key 写入时，应返回 `"updated"` 并覆盖原有内容（而非创建重复条目）
- 验证覆盖后 `content`、`importance`、`tags` 均更新为新值
- 验证记忆总数始终为 1

#### 2. `test_add_memory_deduplicates_same_content`

测试基于内容的去重逻辑（`add_memory`）：

- 第一次写入某内容时，应返回 `True`（新增成功）
- 第二次写入相同内容时，应返回 `False`（跳过，但更新重要性为较大值）
- 验证记忆总数始终为 1，importance 取两次中的最大值

### `test_memory_writer.py`

与 `test_memory_store.py` 内容一致，测试同样的两个用例，不同之处在于使用 `MemoryStore.load()` 而非直接构造 `MemoryStore(path=path, memories=[])`。

## 测试方式

- 基于 Python 标准库 `unittest` 框架
- 使用 `tempfile.TemporaryDirectory` 创建临时目录，确保测试之间文件系统隔离
- 通过 `sys.path.insert` 将项目根目录加入模块搜索路径，直接导入 `memory_store` 模块

### 运行测试

```bash
# 在项目根目录下运行
python -m pytest testcode/ -v

# 或使用 unittest
python -m unittest testcode.test_memory_store -v
python -m unittest testcode.test_memory_writer -v
```
