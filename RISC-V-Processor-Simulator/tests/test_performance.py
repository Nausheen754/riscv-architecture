"""Performance-metric definitions (Stage 2)."""
import pytest

from riscv_engine import (
    clean_source, calculate_comparison, safe_ratio, make_cpu, run_pipeline,
    DEFAULT_PROGRAM,
)


def compare(source):
    return calculate_comparison(clean_source(source))


def test_safe_ratio_handles_zero():
    assert safe_ratio(6, 3) == 2
    assert safe_ratio(5, 0) is None
    assert safe_ratio(0, 4) == 0


def test_hazard_free_program_metrics():
    c = compare("ADDI x1, x0, 1\nADDI x2, x0, 2\nADDI x3, x0, 3\nADDI x4, x0, 4")
    assert c["assembled_count"] == 4 and c["instruction_count"] == 4
    assert c["pipeline_cycles"] == 8 == c["pipe"]["cycles"]
    assert c["ideal_cycles"] == 8 and c["extra_cycles"] == 0
    assert c["serial_five_stage_cycles"] == 20
    assert c["speedup"] == pytest.approx(20 / 8)
    assert c["pipe_cpi"] == pytest.approx(2.0)      # cycles / completed
    assert c["pipe_ipc"] == pytest.approx(0.5)
    assert c["cycles_saved"] == 12
    assert c["cycle_reduction_pct"] == pytest.approx(60.0)
    assert c["mismatches"] == []


def test_counts_come_from_the_executed_program_not_the_static_listing():
    c = compare("ADDI x1, x0, 7\nADDI x2, x0, 7\nBEQ x1, x2, t\n"
                "ADDI x3, x0, 99\nADDI x4, x0, 98\nt: ADDI x5, x0, 1")
    assert c["assembled_count"] == 6                # static program length
    assert c["instruction_count"] == 4              # committed
    assert c["never_executed"] == 2                 # skipped by the branch
    assert c["discarded_count"] == 1                # fetched wrong-path, flushed
    assert c["pipe"]["flushes"] == 1
    assert c["serial_five_stage_cycles"] == 4 * 5   # uses committed, not 6
    assert c["pipeline_cycles"] == 4 + 4 + 2


def test_loop_counts_repeated_execution():
    c = compare("ADDI x1, x0, 3\nloop: ADDI x1, x1, -1\nBEQ x1, x0, done\n"
                "BEQ x0, x0, loop\ndone: ADDI x2, x0, 1")
    assert c["assembled_count"] == 5
    assert c["instruction_count"] > c["assembled_count"] - 1
    assert c["never_executed"] == 0
    assert c["instruction_count"] == c["seq"]["completed"]


def test_stalls_flushes_and_forwarding_are_separate_counters():
    c = compare(DEFAULT_PROGRAM)
    p = c["pipe"]
    assert (p["stalls"], p["flushes"], p["discarded"]) == (1, 1, 1)
    assert p["forwarding"] > 0                      # operands, not cycles
    assert c["extra_cycles"] == p["stalls"] + 2     # one stall + 2 flush bubbles
    # each stall / flush really is recorded in the timeline
    assert sum(r["IF"] == "HELD" for r in p["timeline"]) == p["stalls"]
    assert sum(r["IF"] == "FLUSH" for r in p["timeline"]) == p["flushes"]


def test_forwarding_counts_each_forwarded_operand_once():
    # ADD x2, x1, x1 forwards rs1 and rs2 -> two operand forwards
    p = compare("ADDI x1, x0, 1\nADD x2, x1, x1")["pipe"]
    assert p["forwarding"] == 2
    # a dependency that is satisfied through the register file is not counted
    p = compare("ADDI x1, x0, 1\nADDI x5, x0, 1\nADDI x6, x0, 1\n"
                "ADD x2, x1, x1")["pipe"]
    assert p["forwarding"] == 0


def test_cpi_is_cycles_over_completed_and_none_when_nothing_completed():
    cpu = make_cpu(clean_source("ADDI x1, x0, 1"))
    assert safe_ratio(cpu["cycles"], cpu["completed"]) is None   # not started
    run_pipeline(cpu)
    assert safe_ratio(cpu["cycles"], cpu["completed"]) == 5.0   # 5 cycles / 1


def test_comparison_reports_faults_instead_of_numbers():
    with pytest.raises(RuntimeError, match="fault"):
        compare("ADDI x1, x0, 2\nLW x2, 0(x1)")


def test_speedup_can_be_below_one_and_verdict_says_so():
    # a single instruction: 1 x 5 = 5 serial cycles vs 5 pipeline cycles
    c = compare("ADDI x1, x0, 1")
    assert c["speedup"] == pytest.approx(1.0) and c["cycles_saved"] == 0
    # branch-heavy code: flush bubbles can make the pipeline slower than baseline
    c = compare("BEQ x0, x0, a\na: BEQ x0, x0, b\nb: BEQ x0, x0, c\nc: BEQ x0, x0, d\n"
                "d: BEQ x0, x0, e\ne: BEQ x0, x0, f\nf:")
    assert c["cycles_saved"] == c["serial_five_stage_cycles"] - c["pipeline_cycles"]
    assert ("more cycles" in c["performance_verdict"]) == (c["cycles_saved"] < 0)
