"""Cycle-by-cycle checks of the five-stage pipeline engine."""
import pytest

from riscv_engine import clean_source, make_cpu, pipe_cycle, run_pipeline
from helpers import run_both, assert_agree


def run_pipe(source):
    cpu = make_cpu(clean_source(source))
    run_pipeline(cpu)
    return cpu


def stage_cycles(cpu, text):
    """{stage: [cycle numbers]} in which instruction `text` was shown."""
    found = {s: [] for s in ("IF", "ID", "EX", "MEM", "WB")}
    for row in cpu["timeline"]:
        for stage in found:
            if row[stage] == text:
                found[stage].append(row["Cycle"])
    return found


# 1. independent arithmetic -------------------------------------------------
def test_independent_instructions_fill_and_drain_the_pipeline():
    seq, pipe = run_both("ADDI x1, x0, 1\nADDI x2, x0, 2\nADDI x3, x0, 3")
    assert_agree(seq, pipe)
    assert pipe["cycles"] == 3 + 4                 # n + 4 fill/drain cycles
    assert (pipe["stalls"], pipe["flushes"], pipe["forwarding"]) == (0, 0, 0)
    assert pipe["completed"] == 3
    # exact stage timing of the first instruction
    assert stage_cycles(pipe, "ADDI x1, x0, 1") == {
        "IF": [1], "ID": [2], "EX": [3], "MEM": [4], "WB": [5]}
    assert stage_cycles(pipe, "ADDI x3, x0, 3")["WB"] == [7]


# 2. immediate dependency ---------------------------------------------------
def test_back_to_back_dependency_is_forwarded_without_stall():
    seq, pipe = run_both("ADDI x1, x0, 5\nADD x2, x1, x1")
    assert_agree(seq, pipe)
    assert pipe["regs"][2] == 10
    assert pipe["cycles"] == 6 and pipe["stalls"] == 0
    assert pipe["forwarding"] == 2                  # rs1 and rs2 from EX/MEM


# 3 + 4. several writers of one register, newest must win ------------------------
def test_newest_of_several_writers_is_forwarded():
    seq, pipe = run_both("ADDI x1, x0, 1\nADDI x1, x0, 2\nADDI x1, x0, 3\n"
                         "ADD x2, x1, x1")
    assert_agree(seq, pipe)
    assert pipe["regs"][1] == 3 and pipe["regs"][2] == 6
    assert pipe["forwarding"] == 2                  # only the EX/MEM copies


def test_ex_mem_beats_mem_wb_for_the_same_register():
    seq, pipe = run_both("ADDI x1, x0, 1\nADDI x1, x0, 2\nADD x2, x1, x1")
    assert pipe["regs"][2] == 4                     # not 2 (the older value 1)


def test_mem_wb_forwarding_and_register_file_bypass():
    # distance 2: forwarded from MEM/WB
    seq, pipe = run_both("ADDI x1, x0, 7\nADDI x5, x0, 1\nADD x2, x1, x1")
    assert_agree(seq, pipe)
    assert pipe["regs"][2] == 14 and pipe["forwarding"] == 2
    # distance 3: no forwarding; WB writes the register before ID reads it
    seq, pipe = run_both("ADDI x1, x0, 7\nADDI x5, x0, 1\nADDI x6, x0, 1\n"
                         "ADD x2, x1, x1")
    assert_agree(seq, pipe)
    assert pipe["regs"][2] == 14 and pipe["forwarding"] == 0
    assert pipe["cycles"] == 4 + 4


def test_only_the_dependent_operand_is_forwarded():
    seq, pipe = run_both("ADDI x1, x0, 3\nADDI x2, x0, 4\nADDI x9, x0, 0\n"
                         "ADDI x9, x0, 0\nADDI x9, x0, 0\nADD x3, x1, x9\n"
                         "SUB x4, x3, x2")
    assert_agree(seq, pipe)
    assert (pipe["regs"][3], pipe["regs"][4]) == (3, -1)


# 5. load-use hazard ------------------------------------------------------------
LOAD_USE = ("ADDI x1, x0, 100\nADDI x2, x0, 77\nSW x2, 0(x1)\n"
            "ADDI x2, x0, 5\nLW x2, 0(x1)\nADD x3, x2, x2")


def test_load_use_inserts_exactly_one_stall_and_uses_loaded_value():
    seq, pipe = run_both(LOAD_USE)
    assert_agree(seq, pipe)
    # x2 held 5 before the load, so a stale read would give 10, not 154
    assert pipe["regs"][2] == 77 and pipe["regs"][3] == 154
    assert pipe["stalls"] == 1
    assert pipe["cycles"] == 6 + 4 + 1
    assert pipe["completed"] == 6                   # nothing lost or duplicated


def test_load_use_timeline_shows_hold_and_bubble():
    pipe = run_pipe(LOAD_USE)
    add = stage_cycles(pipe, "ADD x3, x2, x2")
    assert len(add["ID"]) == 2                      # held in ID for 2 cycles
    assert add["ID"][1] == add["ID"][0] + 1
    assert add["EX"] == [add["ID"][1] + 1]          # then straight to EX
    held = [r for r in pipe["timeline"] if r["IF"] == "HELD"]
    assert len(held) == 1 and held[0]["Cycle"] == add["ID"][0]
    # the bubble: the cycle after the hold has an empty EX stage
    bubble_row = pipe["timeline"][held[0]["Cycle"]]   # next cycle (0-indexed)
    assert bubble_row["EX"] == "-"
    assert any("Load-use stall" in e["Events"] for e in pipe["events"])


@pytest.mark.parametrize("consumer", [
    "ADD x3, x2, x0",          # uses rs1
    "ADD x3, x0, x2",          # uses rs2
    "ADDI x3, x2, 1",          # I-type
    "SW x3, 0(x2)",            # load result is the store BASE address
    "BEQ x2, x0, 4",           # branch operand
])
def test_load_use_detected_for_every_consumer_form(consumer):
    seq, pipe = run_both("ADDI x2, x0, 8\nSW x2, 0(x0)\nLW x2, 0(x0)\n" + consumer)
    assert_agree(seq, pipe)
    assert pipe["stalls"] == 1


def test_no_stall_when_load_result_is_not_used_next():
    # LW at index 2, consumer at index 4: one independent instruction between
    seq, pipe = run_both("ADDI x2, x0, 8\nSW x2, 0(x0)\nLW x3, 0(x0)\n"
                         "ADDI x5, x0, 1\nADD x4, x3, x3")
    assert_agree(seq, pipe)
    assert pipe["stalls"] == 0 and pipe["regs"][4] == 16
    assert pipe["cycles"] == 5 + 4              # no hazards: n + 4


def test_load_into_x0_never_stalls():
    seq, pipe = run_both("ADDI x1, x0, 4\nSW x1, 0(x0)\nLW x0, 0(x0)\nADD x2, x0, x0")
    assert_agree(seq, pipe)
    assert pipe["stalls"] == 0 and pipe["regs"][2] == 0


# 6. load followed by dependent store -------------------------------------------
def test_load_then_store_of_loaded_value():
    seq, pipe = run_both("ADDI x1, x0, 100\nADDI x2, x0, 55\nSW x2, 0(x1)\n"
                         "LW x3, 0(x1)\nSW x3, 4(x1)\nLW x4, 4(x1)")
    assert_agree(seq, pipe)
    assert pipe["mem"] == {100: 55, 104: 55}
    assert (pipe["regs"][3], pipe["regs"][4]) == (55, 55)
    assert pipe["stalls"] == 1 and pipe["cycles"] == 6 + 4 + 1


# 7. store depending on recently produced values -------------------------------
def test_store_uses_forwarded_data_and_address():
    seq, pipe = run_both("ADDI x1, x0, 100\nADDI x2, x0, 9\nSW x2, 0(x1)")
    assert_agree(seq, pipe)
    assert pipe["mem"] == {100: 9}
    assert pipe["forwarding"] == 2                  # x2 from EX/MEM, x1 from MEM/WB
    assert pipe["stalls"] == 0


def test_store_data_forwarded_from_older_writer_not_stale_one():
    seq, pipe = run_both("ADDI x2, x0, 1\nADDI x2, x0, 2\nSW x2, 0(x0)")
    assert pipe["mem"] == {0: 2}


# 8 + 9. taken / not-taken branch -----------------------------------------------
def test_taken_branch_flushes_and_skips_wrong_path():
    seq, pipe = run_both("ADDI x1, x0, 7\nADDI x2, x0, 7\nBEQ x1, x2, target\n"
                         "ADDI x3, x0, 99\ntarget: ADDI x4, x0, 1")
    assert_agree(seq, pipe)
    assert (pipe["regs"][3], pipe["regs"][4]) == (0, 1)
    assert pipe["flushes"] == 1 and pipe["discarded"] == 1
    assert pipe["completed"] == 4 == seq["completed"]
    assert pipe["cycles"] == 10                     # 4 + 4 fill/drain + 2 bubbles
    flush_rows = [r for r in pipe["timeline"] if r["IF"] == "FLUSH"]
    assert len(flush_rows) == 1
    assert any("Taken branch" in e["Events"] for e in pipe["events"])


def test_not_taken_branch_has_no_penalty():
    seq, pipe = run_both("ADDI x1, x0, 7\nADDI x2, x0, 8\nBEQ x1, x2, target\n"
                         "ADDI x3, x0, 5\ntarget: ADDI x4, x0, 1")
    assert_agree(seq, pipe)
    assert (pipe["regs"][3], pipe["regs"][4]) == (5, 1)
    assert (pipe["flushes"], pipe["discarded"], pipe["stalls"]) == (0, 0, 0)
    assert pipe["cycles"] == 5 + 4


# 10. wrong-path instructions must never commit ---------------------------------
def test_wrong_path_store_and_arithmetic_never_commit():
    pipe = run_pipe("ADDI x1, x0, 5\nBEQ x0, x0, end\nSW x1, 0(x0)\n"
                    "ADDI x2, x0, 9\nADDI x3, x0, 9\nend: ADDI x4, x0, 1")
    assert pipe["mem"] == {}
    assert (pipe["regs"][2], pipe["regs"][3], pipe["regs"][4]) == (0, 0, 1)
    for row in pipe["timeline"]:
        for stage in ("EX", "MEM", "WB"):
            assert row[stage] not in ("SW x1, 0(x0)", "ADDI x2, x0, 9")
    assert pipe["completed"] == 3                   # ADDI, BEQ, final ADDI


def test_taken_branch_to_end_of_program_drains_cleanly():
    seq, pipe = run_both("ADDI x1, x0, 1\nBEQ x0, x0, end\nADDI x2, x0, 2\nend:")
    assert_agree(seq, pipe)
    assert pipe["halted"] and pipe["regs"][2] == 0 and pipe["completed"] == 2


def test_back_to_back_taken_branches():
    seq, pipe = run_both("BEQ x0, x0, a\nADDI x1, x0, 1\na: BEQ x0, x0, b\n"
                         "ADDI x2, x0, 2\nb: ADDI x3, x0, 3")
    assert_agree(seq, pipe)
    assert (pipe["regs"][1], pipe["regs"][2], pipe["regs"][3]) == (0, 0, 3)
    assert pipe["flushes"] == 2


def test_backward_branch_loop_matches_sequential():
    seq, pipe = run_both("ADDI x1, x0, 4\nADDI x2, x0, 0\n"
                         "loop: ADDI x2, x2, 3\nADDI x1, x1, -1\n"
                         "BEQ x1, x0, done\nBEQ x0, x0, loop\ndone: ADDI x3, x0, 1")
    assert_agree(seq, pipe)
    assert pipe["regs"][2] == 12 and pipe["flushes"] == 4


# 11. branches depending on fresh values --------------------------------------------
@pytest.mark.parametrize("second, expect_x3", [(4, 0), (5, 1)])
def test_branch_operands_are_forwarded(second, expect_x3):
    seq, pipe = run_both(f"ADDI x2, x0, 4\nADDI x1, x0, {second}\n"
                         "BEQ x1, x2, skip\nADDI x3, x0, 1\nskip: ADDI x4, x0, 1")
    assert_agree(seq, pipe)
    assert pipe["regs"][3] == expect_x3 and pipe["regs"][4] == 1


def test_branch_on_value_just_loaded():
    seq, pipe = run_both("ADDI x2, x0, 8\nSW x2, 0(x0)\nLW x1, 0(x0)\n"
                         "BEQ x1, x2, skip\nADDI x3, x0, 1\nskip: ADDI x4, x0, 1")
    assert_agree(seq, pipe)
    assert pipe["regs"][3] == 0 and pipe["stalls"] == 1


def test_branch_uses_newest_of_two_writers():
    seq, pipe = run_both("ADDI x1, x0, 1\nADDI x1, x0, 2\nBEQ x1, x0, t\n"
                         "ADDI x3, x0, 9\nt: ADDI x4, x0, 6")
    assert (pipe["regs"][3], pipe["regs"][4]) == (9, 6)       # 2 != 0: not taken


# 12 + 13. tiny and empty programs ------------------------------------------------
def test_single_instruction_program():
    seq, pipe = run_both("ADDI x5, x0, 5")
    assert_agree(seq, pipe)
    assert pipe["cycles"] == 5 and pipe["completed"] == 1 and pipe["regs"][5] == 5


def test_empty_program_is_halted_immediately():
    cpu = make_cpu(clean_source("# nothing"))
    assert cpu["halted"]
    pipe_cycle(cpu)
    assert cpu["cycles"] == 0 and cpu["timeline"] == []


# 14. completion and halting --------------------------------------------------------
def test_pipeline_halts_once_and_extra_cycles_do_nothing():
    cpu = make_cpu(clean_source("ADDI x1, x0, 1\nADDI x2, x0, 2"))
    while not cpu["halted"]:
        pipe_cycle(cpu)
    snapshot = (cpu["cycles"], cpu["completed"], list(cpu["regs"]),
                len(cpu["timeline"]), len(cpu["events"]))
    for _ in range(5):
        pipe_cycle(cpu)
    assert snapshot == (cpu["cycles"], cpu["completed"], list(cpu["regs"]),
                        len(cpu["timeline"]), len(cpu["events"]))
    assert all(v is None for v in cpu["pipe"].values())
    assert cpu["timeline"][-1]["WB"] == "ADDI x2, x0, 2"


def test_infinite_loop_is_stopped_by_the_cycle_limit():
    cpu = make_cpu(clean_source("loop: BEQ x0, x0, loop"))
    with pytest.raises(RuntimeError, match="Stopped after 40 cycles"):
        run_pipeline(cpu, max_cycles=40)
    assert cpu["cycles"] == 40 and not cpu["halted"]


# 15. x0 through every stage -----------------------------------------------------------
def test_x0_stays_zero_in_every_cycle():
    cpu = make_cpu(clean_source(
        "ADDI x0, x0, 99\nADD x1, x0, x0\nADDI x2, x0, 8\nSW x2, 0(x0)\n"
        "LW x0, 0(x0)\nADD x3, x0, x2\nSUB x0, x2, x2\nOR x4, x0, x2"))
    while not cpu["halted"]:
        pipe_cycle(cpu)
        assert cpu["regs"][0] == 0
    assert cpu["regs"][1] == 0 and cpu["regs"][3] == 8 and cpu["regs"][4] == 8


def test_writes_to_x0_are_never_forwarded():
    pipe = run_pipe("ADDI x0, x0, 5\nADD x1, x0, x0")
    assert pipe["regs"][1] == 0 and pipe["forwarding"] == 0


# invariants ------------------------------------------------------------------------------
def test_every_instruction_flows_through_the_stages_in_order():
    pipe = run_pipe(LOAD_USE)
    for inst in pipe["program"]:
        s = stage_cycles(pipe, inst["text"])
        assert len(s["IF"]) == 1 and len(s["EX"]) == 1
        assert len(s["MEM"]) == 1 and len(s["WB"]) == 1
        assert s["IF"][0] + 1 == s["ID"][0]
        assert s["ID"][-1] + 1 == s["EX"][0]
        assert s["EX"][0] + 1 == s["MEM"][0] and s["MEM"][0] + 1 == s["WB"][0]


def test_stage_registers_match_timeline_after_each_cycle():
    cpu = make_cpu(clean_source(LOAD_USE))
    while not cpu["halted"]:
        pipe_cycle(cpu)
        assert cpu["timeline"][-1]["Cycle"] == cpu["cycles"] == len(cpu["timeline"])
        assert cpu["events"][-1]["Cycle"] == cpu["cycles"]


# faults ---------------------------------------------------------------------------------------
def test_memory_fault_halts_pipeline_precisely():
    cpu = make_cpu(clean_source("ADDI x1, x0, 5\nADDI x2, x0, 2\nLW x3, 0(x2)\n"
                                "ADDI x4, x0, 4"))
    for _ in range(30):
        pipe_cycle(cpu)
    assert cpu["halted"] and "unaligned" in cpu["fault"]
    assert "LW x3, 0(x2)" in cpu["fault"]
    assert cpu["completed"] == 2                    # the two older ADDIs only
    assert cpu["regs"][3] == 0 and cpu["regs"][4] == 0
    assert cpu["cycles"] == 6                       # extra Step clicks add nothing
    assert cpu["timeline"][-1]["MEM"].startswith("FAULT")
    assert all(v is None for v in cpu["pipe"].values())
    with pytest.raises(RuntimeError, match="fault"):
        run_pipeline(cpu)


def test_fault_state_matches_sequential_cpu():
    from riscv_engine import sequential_step
    source = ("ADDI x1, x0, 5\nADDI x2, x0, 2\nSW x1, 0(x0)\nLW x3, 0(x2)\n"
              "ADDI x4, x0, 4")
    pipe = make_cpu(clean_source(source))
    seq = make_cpu(clean_source(source))
    for _ in range(30):
        pipe_cycle(pipe)
        sequential_step(seq)
    assert pipe["fault"] and seq["fault"]
    assert pipe["regs"] == seq["regs"] and pipe["mem"] == seq["mem"]
    assert pipe["completed"] == seq["completed"] == 3
