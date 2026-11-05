"""Small helpers shared by the test files."""
from riscv_engine import (
    clean_source, make_cpu, run_sequential_to_completion, run_pipeline,
    compare_cpu_states,
)


def run_both(source):
    """Assemble `source`, run it on both CPUs, return (seq_cpu, pipe_cpu)."""
    program = clean_source(source)
    seq, pipe = make_cpu(program), make_cpu(program)
    run_sequential_to_completion(seq)
    run_pipeline(pipe)
    return seq, pipe


def assert_agree(seq, pipe):
    """Sequential and pipelined CPUs must reach the same architectural state."""
    assert compare_cpu_states(seq, pipe) == []
    assert seq["regs"][0] == 0 and pipe["regs"][0] == 0
