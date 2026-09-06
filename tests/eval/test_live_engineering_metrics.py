import json

from langchain_core.messages import AIMessage

from tests.eval.live_engineering_metrics import (
    PLAN_CASES,
    ROUTE_CASES,
    LOOP_CASES,
    score_plan_text,
    score_route_message,
    wilson_interval,
)


def test_live_datasets_have_100_distinct_cases():
    all_ids = [case["id"] for case in ROUTE_CASES + PLAN_CASES + LOOP_CASES]
    assert len(ROUTE_CASES) == 50
    assert len(PLAN_CASES) == 25
    assert len(LOOP_CASES) == 25
    assert len(set(all_ids)) == 100


def test_route_score_requires_name_schema_and_exact_args():
    case = ROUTE_CASES[0]
    good = AIMessage(content="", tool_calls=[{"name": case["tool"], "args": case["args"], "id": "c1"}])
    wrong_args = AIMessage(content="", tool_calls=[{"name": case["tool"], "args": {"city": "北京"}, "id": "c2"}])
    assert score_route_message(good, case)["passed"] is True
    assert score_route_message(wrong_args, case)["passed"] is False


def test_plan_score_requires_strict_json_coverage_and_order():
    case = PLAN_CASES[0]
    raw = json.dumps([f"读取 {case['anchors'][0]}", f"应用 {case['anchors'][1]}", f"写入 {case['anchors'][2]}"], ensure_ascii=False)
    reversed_raw = json.dumps([f"写入 {case['anchors'][2]}", f"应用 {case['anchors'][1]}", f"读取 {case['anchors'][0]}"], ensure_ascii=False)
    assert score_plan_text(raw, case)["passed"] is True
    assert score_plan_text(reversed_raw, case)["passed"] is False


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(50, 50)
    assert 0.92 < low < 1.0
    assert high == 1.0
