from __future__ import annotations

from collections import Counter

from tests.eval.subagent_parallel_benchmark import _evaluate, build_cases, summarize


def test_parallel_benchmark_case_matrix_is_balanced_and_allows_autonomous_delegation():
    cases = build_cases(2)
    counts = Counter(case["family"] for case in cases)
    assert counts == {
        "release_readiness_review": 2,
        "incident_diagnosis": 2,
        "change_impact_analysis": 2,
        "simple_lookup_control": 2,
    }
    for case in cases:
        assert all(token not in case["prompt"] for token in case["expected_tokens"])
        assert all(
            candidate not in case["prompt"]
            for alternatives in case["expected_any"]
            for candidate in alternatives
        )
        assert "delegate_task" not in case["prompt"]
        assert "委派" not in case["prompt"]
        if case["family"] == "simple_lookup_control":
            assert case["delegation_policy"] == "avoid"
        else:
            assert case["delegation_policy"] == "optional"


def _record(case_id: str, family: str, arm: str, elapsed: float) -> dict:
    return {
        "case_id": case_id,
        "family": family,
        "parallelizable": family != "simple_lookup_control",
        "arm": arm,
        "repeat": 0,
        "result": {"correct": True, "evidence_correct": True},
        "qualified_success": True,
        "delegated": arm == "hierarchical",
        "delegation_appropriate": True,
        "elapsed_seconds": elapsed,
        "parent_wait_fraction": 0.5 if arm == "hierarchical" else 0.0,
        "tool_max_concurrency": 2 if arm == "hierarchical" else 1,
        "llm_calls": 2,
        "input_tokens": 100,
        "output_tokens": 20,
    }


def test_parallel_benchmark_summary_reports_paired_speedup():
    records = [
        _record("p1", "release_readiness_review", "single_agent", 10.0),
        _record("p1", "release_readiness_review", "hierarchical", 5.0),
        _record("s1", "simple_lookup_control", "single_agent", 4.0),
        _record("s1", "simple_lookup_control", "hierarchical", 5.0),
    ]
    result = summarize(records, "other", "test-model")
    assert result["paired"]["n"] == 2
    assert result["paired"]["parallel_n"] == 1
    assert result["paired"]["parallel_mean_speedup"] == 2.0
    assert result["paired"]["parallel_hierarchical_faster_rate"] == 1.0
    assert result["paired"]["actually_delegated_n"] == 2
    assert result["paired"]["actually_delegated_input_token_ratio"] == 1.0
    assert result["autonomy"]["complex_delegation_rate"] == 1.0
    assert result["autonomy"]["simple_overdelegation_rate"] == 1.0


def test_release_verdict_accepts_natural_blocking_wording():
    case = build_cases(1)[0]
    text = "不得上线 " + " ".join(
        (*case["expected_tokens"], *case["required_reads"])
    )
    calls = [
        {"name": "read_office_file", "args": {"filepath": path}}
        for path in case["required_reads"]
    ]

    result = _evaluate(case, text, calls)

    assert result["correct"] is True
