"""Sequential vs pipelined CPU on many programs, plus an independent oracle.

The oracle below is a tiny reference interpreter written separately from the
engine (plain Python ints, explicit 32-bit wrap), so the sequential CPU is
not just compared with itself.
"""
import random

import pytest

from riscv_engine import (
    clean_source, make_cpu, run_sequential_to_completion, run_pipeline,
    compare_cpu_states, DEFAULT_PROGRAM,
)
from helpers import run_both, assert_agree


def wrap(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


def oracle(program, limit=5000):
    regs, mem, pc, done = [0] * 32, {}, 0, 0
    while 0 <= pc // 4 < len(program) and done < limit:
        i = program[pc // 4]
        op, a, b = i["op"], regs[i["rs1"]], regs[i["rs2"]]
        nxt = pc + 4
        if op == "ADD": regs[i["rd"]] = wrap(a + b)
        elif op == "SUB": regs[i["rd"]] = wrap(a - b)
        elif op == "AND": regs[i["rd"]] = wrap(a & b)
        elif op == "OR": regs[i["rd"]] = wrap(a | b)
        elif op == "ADDI": regs[i["rd"]] = wrap(a + i["imm"])
        elif op == "LW": regs[i["rd"]] = wrap(mem.get(wrap(a + i["imm"]), 0))
        elif op == "SW": mem[wrap(a + i["imm"])] = b & 0xFFFFFFFF
        elif op == "BEQ" and a == b: nxt = pc + i["imm"]
        regs[0] = 0
        pc, done = nxt, done + 1
    return regs, mem, pc, done


def random_program(rng):
    """Valid, terminating program: forward branches only, aligned memory."""
    n = rng.randint(1, 18)
    lines = []
    for idx in range(n):
        r = lambda: f"x{rng.randint(0, 6)}"
        kind = rng.choice(["alu", "alu", "addi", "addi", "lw", "sw", "beq"])
        if kind == "alu":
            lines.append(f"{rng.choice(['ADD', 'SUB', 'AND', 'OR'])} {r()}, {r()}, {r()}")
        elif kind == "addi":
            lines.append(f"ADDI {r()}, {r()}, {rng.choice([-2, -1, 0, 1, 2, 5, 2047, -2048])}")
        elif kind == "lw":
            lines.append(f"LW {r()}, {rng.randrange(0, 32, 4)}(x0)")
        elif kind == "sw":
            lines.append(f"SW {r()}, {rng.randrange(0, 32, 4)}(x0)")
        else:
            lines.append(f"BEQ {r()}, {r()}, {4 * rng.randint(1, 4)}")
    return "\n".join(lines)


@pytest.mark.parametrize("seed", range(400))
def test_random_programs_agree_with_oracle_and_each_other(seed):
    source = random_program(random.Random(seed))
    program = clean_source(source)
    seq, pipe = make_cpu(program), make_cpu(program)
    run_sequential_to_completion(seq)
    run_pipeline(pipe)

    # 1. sequential and pipelined agree on architectural state
    assert compare_cpu_states(seq, pipe) == [], source
    # 2. both agree with the independent reference interpreter
    regs, mem, pc, done = oracle(program)
    assert seq["regs"] == regs and pipe["regs"] == regs, source
    assert seq["mem"] == mem and pipe["mem"] == mem, source
    assert seq["completed"] == pipe["completed"] == done, source
    assert seq["pc"] == pipe["pc"] == pc, source
    assert pipe["regs"][0] == 0
    # 3. cycle accounting: n + 4 fill/drain + 1 per stall, plus 0..2 bubbles
    #    per taken branch (0 when the branch has nothing left to fetch)
    extra = pipe["cycles"] - (pipe["completed"] + 4) - pipe["stalls"]
    assert 0 <= extra <= 2 * pipe["flushes"], (source, extra)
    assert pipe["discarded"] <= pipe["flushes"]
    assert len(pipe["timeline"]) == pipe["cycles"]


def test_default_example_program():
    seq, pipe = run_both(DEFAULT_PROGRAM)
    assert_agree(seq, pipe)
    r = pipe["regs"]
    assert (r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) == \
        (10, 5, 15, 5, 0, 15, 100, 15)
    assert (r[9], r[10]) == (0, 1)                  # ADDI x9 skipped by BEQ
    assert pipe["mem"] == {100: 15}
    assert seq["completed"] == pipe["completed"] == 11
    assert pipe["stalls"] == 1 and pipe["flushes"] == 1 and pipe["discarded"] == 1
    assert pipe["cycles"] == 11 + 4 + 1 + 2


def test_every_register_and_memory_word_is_compared():
    seq, pipe = run_both("ADDI x1, x0, 1\nSW x1, 0(x0)")
    pipe["regs"][31] = 5
    pipe["mem"][8] = 3
    names = {m["Component"] for m in compare_cpu_states(seq, pipe)}
    assert names == {"x31", "Memory[8]"}
    pipe["completed"] += 1
    pipe["pc"] += 4
    names = {m["Component"] for m in compare_cpu_states(seq, pipe)}
    assert {"Instructions completed", "Final PC"} <= names
