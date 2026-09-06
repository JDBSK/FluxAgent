from tests.eval.resume_metrics import benchmark_watermark, run_memory_comparison


def test_twenty_ten_reduces_summary_frequency_in_sixty_turn_replay():
    report = run_memory_comparison(total_turns=60)

    assert report["baseline"]["summary_calls"] == 14
    assert report["candidate"]["summary_calls"] == 5
    assert report["summary_call_reduction_vs_8_4"] == 0.6429


def test_watermark_replay_keeps_configured_recent_turns():
    eight_four = benchmark_watermark(60, 8, 4)
    twenty_ten = benchmark_watermark(60, 20, 10)

    assert eight_four["raw_turns_at_end"] == 4
    assert twenty_ten["raw_turns_at_end"] == 10
    assert twenty_ten["compressed_turns_total"] == 50
