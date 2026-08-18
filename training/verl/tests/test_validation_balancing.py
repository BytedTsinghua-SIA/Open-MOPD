import ast
from pathlib import Path


def test_validation_keeps_official_interleaved_repeat_order():
    source_path = Path(__file__).parents[1] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    module = ast.parse(source_path.read_text())
    validate = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_validate"
    )

    calls = [node for node in ast.walk(validate) if isinstance(node, ast.Call)]
    repeat_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "repeat"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "test_batch"
    ]
    assert repeat_calls
    assert any(
        any(
            keyword.arg == "interleave"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in repeat_calls
    )

    assert not any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "reorder"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "test_batch"
        for call in calls
    )
