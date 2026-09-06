"""pytest 入口：让 447 条评测能跟普通单测一起跑。

    python -m pytest tests/eval -q

用 parametrize 而不是在一个 test 里 for 循环：失败时 pytest 直接报出
是哪条 case 挂了，不用翻日志。安全类 case 单独聚合成一个严格断言——
安全通过率不允许打折，99% 意味着有一条越权没拦住。
"""
import pytest

from .cases import all_cases
from .runners import run_case

SUITES = all_cases()
CRITICAL_KINDS = {"shell", "path", "gate", "multi_gate", "mcp_call", "mcp_blockflag", "hard_deny"}


def _params(suite_name: str):
    return [pytest.param(c, id=c["id"]) for c in SUITES[suite_name]]


@pytest.mark.parametrize("case", _params("hitl"))
def test_hitl(case):
    passed, detail = run_case(case)
    assert passed, detail


@pytest.mark.parametrize("case", _params("task_state"))
def test_task_state(case):
    passed, detail = run_case(case)
    assert passed, detail


@pytest.mark.parametrize("case", _params("tools"))
def test_tools(case):
    passed, detail = run_case(case)
    assert passed, detail


@pytest.mark.parametrize("case", _params("mcp"))
def test_mcp(case):
    passed, detail = run_case(case)
    assert passed, detail

@pytest.mark.parametrize("case", _params("runtime"))
def test_runtime(case):
    passed, detail = run_case(case)
    assert passed, detail



def test_suite_sizes():
    """每个能力至少 100 条样例——低于这个数就只是 demo 级验证。"""
    for name, cases in SUITES.items():
        assert len(cases) >= 100, f"{name} only has {len(cases)} cases (need >=100)"


def test_case_ids_unique_and_stable():
    """ID 必须唯一且跨进程稳定，否则回归基线对不上。"""
    ids = [c["id"] for cases in SUITES.values() for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    # 不能含内置 hash 的痕迹（PYTHONHASHSEED 会让它每次都变）
    from .cases import _sid
    assert _sid("abc") == _sid("abc")


def test_security_no_partial_credit():
    """安全类断言必须 100%，一条都不能漏。"""
    failures = []
    for cases in SUITES.values():
        for case in cases:
            if case["kind"] not in CRITICAL_KINDS:
                continue
            passed, detail = run_case(case)
            if not passed:
                failures.append(f"{case['id']}: {detail}")
    assert not failures, f"{len(failures)} security failures: {failures[:5]}"
