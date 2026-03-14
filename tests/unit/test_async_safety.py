"""
Static analysis test to catch missing `await` on async function/method calls.

Uses Python's AST to scan all source files and verify that calls to async methods
via `self.method()` are properly awaited. This catches the class of bug where
`get_trend_history()` (async) was called without `await` in temporal_analyzer.py.

The scanner works at the class level: it collects all `async def` methods in a class,
then verifies that `self.method()` calls to those methods within the same class are
awaited. It also tracks cross-class calls via `self.attr.method()` by mapping known
attribute types from `__init__`.
"""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"


def _collect_class_async_methods(tree: ast.Module) -> dict[str, set[str]]:
    """Map class names to their async method names."""
    class_methods: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            async_methods = set()
            for item in ast.walk(node):
                if isinstance(item, ast.AsyncFunctionDef):
                    async_methods.add(item.name)
            if async_methods:
                class_methods[node.name] = async_methods
    return class_methods


def _collect_top_level_async_funcs(tree: ast.Module) -> set[str]:
    """Collect module-level async function names (not methods)."""
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            names.add(node.name)
    return names


def _get_init_attr_types(cls_node: ast.ClassDef) -> dict[str, str]:
    """
    Extract attribute type hints from __init__ to map self.attr to class names.

    Looks for patterns like:
        self.history: DigestHistory = history_db
        self.history = history_db  (with type annotation on parameter)
    """
    attr_types: dict[str, str] = {}
    for item in cls_node.body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if item.name != "__init__":
            continue

        # Map parameter names to their type annotations
        param_types: dict[str, str] = {}
        for arg in item.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                param_types[arg.arg] = arg.annotation.id

        for stmt in ast.walk(item):
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    attr_name = target.attr
                    # Check if RHS is a parameter with a known type
                    if isinstance(stmt.value, ast.Name) and stmt.value.id in param_types:
                        attr_types[attr_name] = param_types[stmt.value.id]

            # Also handle annotated assignments: self.x: Type = value
        for stmt in ast.walk(item):
            if isinstance(stmt, ast.AnnAssign):
                if (
                    isinstance(stmt.target, ast.Attribute)
                    and isinstance(stmt.target.value, ast.Name)
                    and stmt.target.value.id == "self"
                    and isinstance(stmt.annotation, ast.Name)
                ):
                    attr_types[stmt.target.attr] = stmt.annotation.id

    return attr_types


def _collect_awaited_call_ids(func_node: ast.AST) -> set[int]:
    """Collect the Python object IDs of all Call nodes that are awaited."""
    awaited = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            awaited.add(id(node.value))
    return awaited


def _collect_gathered_call_ids(func_node: ast.AST) -> set[int]:
    """Collect Call IDs passed as arguments to asyncio.gather/run/create_task/etc."""
    gathered = set()
    # Async bridge functions that accept coroutines without await
    ASYNC_BRIDGES = {"gather", "create_task", "ensure_future", "run"}

    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ASYNC_BRIDGES:
            # All positional args are coroutines (no await needed)
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    gathered.add(id(arg))
                # Also handle list comprehensions: [coro() for x in y]
                if isinstance(arg, ast.ListComp) and isinstance(arg.elt, ast.Call):
                    gathered.add(id(arg.elt))
            # Also check starargs
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Starred) and isinstance(arg.value, ast.ListComp):
                        if isinstance(arg.value.elt, ast.Call):
                            gathered.add(id(arg.value.elt))
    return gathered


def _check_class_methods(
    cls_node: ast.ClassDef,
    own_async_methods: set[str],
    all_class_methods: dict[str, set[str]],
    attr_types: dict[str, str],
    top_level_async: set[str],
) -> list[tuple[int, str, str]]:
    """Check all methods in a class for unawaited async calls."""
    issues = []

    for method in ast.walk(cls_node):
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        awaited_ids = _collect_awaited_call_ids(method)
        gathered_ids = _collect_gathered_call_ids(method)
        safe_ids = awaited_ids | gathered_ids

        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if id(node) in safe_ids:
                continue

            # Check self.method() calls
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                    method_name = node.func.attr
                    # Direct self.async_method() call
                    if method_name in own_async_methods:
                        issues.append((
                            node.lineno,
                            f"self.{method_name}",
                            f"in {cls_node.name}.{method.name}()",
                        ))

                    # self.attr.async_method() — check if attr maps to a known class
                elif (
                    isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "self"
                ):
                    attr_name = node.func.value.attr
                    method_name = node.func.attr
                    if attr_name in attr_types:
                        target_class = attr_types[attr_name]
                        if target_class in all_class_methods:
                            if method_name in all_class_methods[target_class]:
                                issues.append((
                                    node.lineno,
                                    f"self.{attr_name}.{method_name}",
                                    f"in {cls_node.name}.{method.name}()",
                                ))

            # Check top-level async function calls (not method calls)
            elif isinstance(node.func, ast.Name):
                if node.func.id in top_level_async:
                    issues.append((
                        node.lineno,
                        node.func.id,
                        f"in {cls_node.name}.{method.name}()",
                    ))

    return issues


def _check_top_level_functions(
    tree: ast.Module,
    top_level_async: set[str],
    all_class_methods: dict[str, set[str]],
) -> list[tuple[int, str, str]]:
    """Check module-level functions for unawaited async calls."""
    issues = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        awaited_ids = _collect_awaited_call_ids(node)
        gathered_ids = _collect_gathered_call_ids(node)
        safe_ids = awaited_ids | gathered_ids

        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if id(call) in safe_ids:
                continue

            # asyncio.run() is a sync-to-async bridge — not a missing await
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "asyncio"
                and call.func.attr == "run"
            ):
                continue

            if isinstance(call.func, ast.Name) and call.func.id in top_level_async:
                issues.append((
                    call.lineno,
                    call.func.id,
                    f"in {node.name}()",
                ))

    return issues


def _scan_file(
    filepath: Path,
    global_class_methods: dict[str, set[str]],
    global_top_level_async: set[str],
) -> list[tuple[str, int, str, str]]:
    """Scan a single file for unawaited async calls."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))

    local_class_methods = _collect_class_async_methods(tree)
    local_top_level = _collect_top_level_async_funcs(tree)

    # Merge global + local knowledge
    all_class_methods = {**global_class_methods, **local_class_methods}
    all_top_level = global_top_level_async | local_top_level

    issues = []
    rel_path = str(filepath.relative_to(SRC_DIR.parent))

    # Check each class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            own_async = local_class_methods.get(node.name, set())
            attr_types = _get_init_attr_types(node)
            class_issues = _check_class_methods(
                node, own_async, all_class_methods, attr_types, all_top_level,
            )
            issues.extend((rel_path, line, name, ctx) for line, name, ctx in class_issues)

    # Check top-level functions
    top_issues = _check_top_level_functions(tree, all_top_level, all_class_methods)
    issues.extend((rel_path, line, name, ctx) for line, name, ctx in top_issues)

    return issues


class TestAsyncAwaitSafety:
    """
    Static analysis tests for async/await correctness.

    These tests scan the source code AST to find calls to async methods that
    are missing `await`. The scanner is class-aware: it tracks which methods
    are async on each class, and verifies that `self.method()` and
    `self.attr.method()` calls are properly awaited.
    """

    @pytest.fixture(scope="class")
    def global_class_methods(self) -> dict[str, set[str]]:
        """Collect async methods for all classes across all source files."""
        all_methods: dict[str, set[str]] = {}
        for py_file in SRC_DIR.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            file_methods = _collect_class_async_methods(tree)
            for cls_name, methods in file_methods.items():
                all_methods.setdefault(cls_name, set()).update(methods)
        return all_methods

    @pytest.fixture(scope="class")
    def global_top_level_async(self) -> set[str]:
        """Collect all top-level async function names."""
        names = set()
        for py_file in SRC_DIR.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            names |= _collect_top_level_async_funcs(tree)
        return names

    @pytest.fixture(scope="class")
    def all_issues(
        self, global_class_methods, global_top_level_async,
    ) -> list[tuple[str, int, str, str]]:
        """Scan all source files for unawaited async calls."""
        issues = []
        for py_file in SRC_DIR.rglob("*.py"):
            issues.extend(_scan_file(py_file, global_class_methods, global_top_level_async))
        return issues

    def test_no_unawaited_async_calls(self, all_issues):
        """Verify that no async function/method calls are missing `await`."""
        if all_issues:
            msg_lines = ["Found unawaited async calls:"]
            for filepath, line, name, ctx in all_issues:
                msg_lines.append(f"  {filepath}:{line} - {name}() ({ctx})")
            pytest.fail("\n".join(msg_lines))

    def test_async_method_inventory(self, global_class_methods):
        """
        Sanity check: verify we found async methods to scan.

        If this fails, the scanner isn't finding any async defs and the
        safety test above would vacuously pass.
        """
        total = sum(len(methods) for methods in global_class_methods.values())
        assert total > 10, (
            f"Only found {total} async methods across all classes — "
            f"scanner may be broken"
        )
