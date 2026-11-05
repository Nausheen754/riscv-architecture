import pytest

from riscv_engine import clean_source, make_cpu, sequential_step, \
    run_sequential_to_completion
from helpers import run_both

INT_MAX, INT_MIN = 2**31 - 1, -2**31


def run_seq(source, max_steps=1000):
    cpu = make_cpu(clean_source(source))
    run_sequential_to_completion(cpu, max_steps)
    return cpu


def regs_after(source):
    return run_seq(source)["regs"]


def test_pc_advances_by_four_and_cpu_halts_after_last_instruction():
    cpu = make_cpu(clean_source("ADDI x1, x0, 1\nADDI x2, x0, 2"))
    assert (cpu["pc"], cpu["halted"]) == (0, False)
    sequential_step(cpu)
    assert (cpu["pc"], cpu["halted"], cpu["completed"]) == (4, False, 1)
    sequential_step(cpu)
    assert (cpu["pc"], cpu["halted"], cpu["completed"]) == (8, True, 2)
    sequential_step(cpu)                       # stepping a halted CPU is a no-op
    assert (cpu["pc"], cpu["completed"], cpu["fault"]) == (8, 2, None)


@pytest.mark.parametrize("program, reg, expected", [
    ("ADDI x1, x0, 15\nADDI x2, x0, 5\nADD x3, x1, x2", 3, 20),
    ("ADDI x1, x0, 15\nADDI x2, x0, 5\nSUB x3, x1, x2", 3, 10),
    ("ADDI x1, x0, 5\nADDI x2, x0, 15\nSUB x3, x1, x2", 3, -10),
    ("ADDI x1, x0, 12\nADDI x2, x0, 10\nAND x3, x1, x2", 3, 8),
    ("ADDI x1, x0, 12\nADDI x2, x0, 10\nOR x3, x1, x2", 3, 14),
    ("ADDI x1, x0, -1\nADDI x2, x0, 10\nAND x3, x1, x2", 3, 10),
    ("ADDI x1, x0, -1\nADDI x2, x0, 10\nOR x3, x1, x2", 3, -1),
    ("ADDI x1, x0, -7\nADDI x2, x1, 7", 2, 0),
    ("ADDI x1, x0, 2047", 1, 2047),
    ("ADDI x1, x0, -2048", 1, -2048),
])
def test_alu_instructions(program, reg, expected):
    assert regs_after(program)[reg] == expected


def test_32_bit_overflow_wraps_like_twos_complement():
    cpu = make_cpu(clean_source("ADDI x2, x1, 1\nADD x3, x1, x1\nSUB x4, x5, x1"))
    cpu["regs"][1] = INT_MAX
    cpu["regs"][5] = INT_MIN
    run_sequential_to_completion(cpu)
    assert cpu["regs"][2] == INT_MIN        # INT_MAX + 1 wraps to INT_MIN
    assert cpu["regs"][3] == -2             # INT_MAX + INT_MAX
    assert cpu["regs"][4] == 1              # INT_MIN - INT_MAX


def test_x0_is_always_zero():
    regs = regs_after("ADDI x0, x0, 99\nADD x0, x0, x0\nADDI x1, x0, 7\n"
                      "ADD x2, x0, x1\nLW x0, 0(x0)\nSUB x0, x1, x1")
    assert regs[0] == 0 and regs[1] == 7 and regs[2] == 7


def test_x0_stays_zero_after_load_from_nonzero_memory():
    cpu = run_seq("ADDI x1, x0, 5\nSW x1, 0(x0)\nLW x0, 0(x0)\nADD x2, x0, x0")
    assert cpu["regs"][0] == 0 and cpu["regs"][2] == 0


def test_store_and_load_words():
    cpu = run_seq("ADDI x1, x0, 100\nADDI x2, x0, 77\nSW x2, 0(x1)\n"
                  "SW x2, 8(x1)\nLW x3, 0(x1)\nLW x4, 4(x1)\nLW x5, 8(x1)")
    assert cpu["mem"] == {100: 77, 108: 77}
    assert (cpu["regs"][3], cpu["regs"][4], cpu["regs"][5]) == (77, 0, 77)


def test_negative_values_survive_a_store_load_round_trip():
    # Regression: LW used to return the unsigned word (4294967295 for -1).
    cpu = run_seq("ADDI x1, x0, -1\nSW x1, 0(x0)\nLW x2, 0(x0)")
    assert cpu["mem"][0] == 0xFFFFFFFF            # memory holds the raw word
    assert cpu["regs"][2] == -1


def test_branch_compares_loaded_negative_value_correctly():
    # Regression: BEQ x1, x2 failed when x2 came from LW of the same -1.
    cpu = run_seq("ADDI x1, x0, -1\nSW x1, 0(x0)\nLW x2, 0(x0)\n"
                  "BEQ x1, x2, skip\nADDI x3, x0, 99\nskip: ADDI x4, x0, 1")
    assert cpu["regs"][3] == 0 and cpu["regs"][4] == 1


def test_negative_offset_and_base_register():
    cpu = run_seq("ADDI x1, x0, 108\nADDI x2, x0, 9\nSW x2, -8(x1)\nLW x3, -8(x1)")
    assert cpu["mem"] == {100: 9} and cpu["regs"][3] == 9


def test_uninitialised_memory_reads_zero():
    assert regs_after("ADDI x1, x0, 40\nLW x2, 0(x1)")[2] == 0


@pytest.mark.parametrize("setup, bad", [
    ("ADDI x1, x0, 2", "LW x2, 0(x1)"),           # unaligned load
    ("ADDI x1, x0, 6", "SW x2, 0(x1)"),           # unaligned store
    ("ADDI x1, x0, -4", "LW x2, 0(x1)"),          # negative address
    ("ADDI x1, x0, -8", "SW x2, 4(x1)"),
])
def test_invalid_memory_access_halts_with_a_fault(setup, bad):
    cpu = make_cpu(clean_source(f"{setup}\n{bad}\nADDI x9, x0, 1"))
    for _ in range(10):
        sequential_step(cpu)
    assert cpu["halted"] and cpu["fault"]
    assert "address" in cpu["fault"] and bad.split(",")[0] in cpu["fault"]
    assert cpu["completed"] == 1 and cpu["regs"][9] == 0 and cpu["mem"] == {}
    with pytest.raises(RuntimeError):
        run_sequential_to_completion(make_cpu(clean_source(f"{setup}\n{bad}")))


def test_beq_taken_forward_and_not_taken():
    taken = run_seq("ADDI x1, x0, 3\nADDI x2, x0, 3\nBEQ x1, x2, 8\n"
                    "ADDI x3, x0, 1\nADDI x4, x0, 2")
    assert taken["regs"][3] == 0 and taken["regs"][4] == 2
    assert [h["PC"] for h in taken["history"]] == [0, 4, 8, 16]
    not_taken = run_seq("ADDI x1, x0, 3\nADDI x2, x0, 4\nBEQ x1, x2, 8\n"
                        "ADDI x3, x0, 1\nADDI x4, x0, 2")
    assert not_taken["regs"][3] == 1 and not_taken["regs"][4] == 2


def test_beq_target_is_pc_relative_and_label_offset_matches_number():
    by_label = run_seq("BEQ x0, x0, end\nADDI x1, x0, 1\nADDI x1, x0, 2\n"
                       "end: ADDI x2, x0, 3")
    by_number = run_seq("BEQ x0, x0, 12\nADDI x1, x0, 1\nADDI x1, x0, 2\n"
                        "ADDI x2, x0, 3")
    assert by_label["regs"] == by_number["regs"]
    assert by_label["regs"][1] == 0 and by_label["regs"][2] == 3


def test_backward_branch_loop_counts_down():
    cpu = run_seq("ADDI x1, x0, 3\nADDI x2, x0, 0\n"
                  "loop: ADDI x2, x2, 10\nADDI x1, x1, -1\nBEQ x1, x0, done\n"
                  "BEQ x0, x0, loop\ndone: ADDI x3, x0, 1")
    assert cpu["regs"][2] == 30 and cpu["regs"][1] == 0 and cpu["regs"][3] == 1


def test_branch_past_end_of_program_halts():
    cpu = run_seq("BEQ x0, x0, 400\nADDI x1, x0, 1")
    assert cpu["halted"] and cpu["regs"][1] == 0 and cpu["completed"] == 1


def test_infinite_loop_is_stopped_by_the_step_limit():
    with pytest.raises(RuntimeError, match="stopped after 50 instructions"):
        run_seq("loop: BEQ x0, x0, loop", max_steps=50)


def test_single_instruction_program():
    cpu = run_seq("ADDI x5, x0, 5")
    assert cpu["regs"][5] == 5 and cpu["halted"] and cpu["completed"] == 1
