import ast
from pathlib import Path


def _reconstruct_tree():
    return ast.parse((Path(__file__).parents[1] / "reconstruct.py").read_text())


def test_empty_cache_policy_defaults_on_and_is_resolved_before_carve_loop():
    tree = _reconstruct_tree()
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "empty_cache_each_iter" for target in node.targets)
    ]
    assert len(assignments) == 1
    call = assignments[0].value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name) and call.func.id == "_env_bool"
    assert ast.literal_eval(call.args[0]) == "PSHUMAN_EMPTY_CACHE_EACH_ITER"
    assert ast.literal_eval(call.args[1]) is True

    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "i"
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "empty_cache"
            for child in ast.walk(node)
        )
    ]
    assert len(loops) == 1
    assert assignments[0].lineno < loops[0].lineno


def test_only_empty_cache_call_is_guarded_by_the_policy():
    tree = _reconstruct_tree()
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not isinstance(node.test, ast.Name) or node.test.id != "empty_cache_each_iter":
            continue
        calls = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "empty_cache"
        ]
        guarded.extend(calls)
    assert len(guarded) == 1


def test_runtime_counter_is_incremented_only_with_the_guarded_call():
    tree = _reconstruct_tree()
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "empty_cache_each_iter"
    ]
    assert len(guards) == 1
    increments = [
        node
        for node in ast.walk(guards[0])
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "empty_cache_calls"
        and isinstance(node.op, ast.Add)
        and ast.literal_eval(node.value) == 1
    ]
    assert len(increments) == 1

    emits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "emit"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "pshuman: carve optimizer loop (with remesh)"
    ]
    assert len(emits) == 1
    keywords = {keyword.arg: keyword.value for keyword in emits[0].keywords if keyword.arg}
    assert isinstance(keywords["empty_cache_calls"], ast.Name)
    assert keywords["empty_cache_calls"].id == "empty_cache_calls"
