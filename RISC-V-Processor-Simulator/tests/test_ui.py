"""Headless checks of the Streamlit interface using streamlit's AppTest."""
import os
import warnings

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest      # noqa: E402

from riscv_engine import DEFAULT_PROGRAM       # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
SEQ, PIPE, MACHINE, PERF, TESTS = 1, 2, 3, 4, 5      # tab positions


def new_app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def metrics(at, tab):
    return {m.label: m.value for m in at.tabs[tab].metric}


def assemble(at, source=None):
    if source is not None:
        at.text_area(key="assembly_editor").set_value(source)
    next(b for b in at.button if b.label == "Assemble Program").click().run()
    return at


def texts(elements):
    return [e.value for e in elements]


def test_app_starts_without_errors_and_shows_default_program():
    at = new_app()
    assert at.text_area(key="assembly_editor").value == DEFAULT_PROGRAM
    assert any("Assemble a program first" in v for v in texts(at.tabs[SEQ].info))


def test_assemble_default_program():
    at = assemble(new_app())
    assert "Successfully assembled 12 instructions." in texts(at.success)
    assert not at.exception


def test_empty_program_gives_a_clear_message():
    at = assemble(new_app(), "# only a comment\n\n")
    assert any("Program is empty" in v for v in texts(at.error))


@pytest.mark.parametrize("source, fragment", [
    ("ADD x1, x2", "Expected rd, rs1, rs2"),
    ("LW x1, 4, x2", "offset(base)"),
    ("FOO x1", "Unsupported instruction: FOO"),
    (",", "Unsupported instruction"),
])
def test_assembly_errors_are_shown_not_tracebacks(source, fragment):
    at = assemble(new_app(), source)
    assert not at.exception
    assert any(fragment in v for v in texts(at.error))


def test_load_default_example_restores_the_editor_text():
    # Regression: the editor kept the user's text after 'Load Default Example'.
    at = assemble(new_app(), "ADDI x1, x0, 1")
    at.sidebar.button[0].click().run()
    assert at.text_area(key="assembly_editor").value == DEFAULT_PROGRAM
    assert at.session_state["program"] == []              # nothing stale
    assert any("Assemble a program first" in v for v in texts(at.tabs[SEQ].info))


def test_editing_after_assembly_shows_stale_warning_on_every_result_tab():
    at = assemble(new_app())
    assert not any("changed since" in v for v in texts(at.warning))
    at.text_area(key="assembly_editor").set_value("ADDI x9, x0, 9").run()
    for tab in (SEQ, PIPE, PERF):
        assert any("previously assembled" in v for v in texts(at.tabs[tab].warning)), tab
    # re-assembling clears the warning and replaces the old results
    assemble(at)
    assert not any("previously assembled" in v for v in texts(at.warning))
    assert at.session_state["program"][0]["text"] == "ADDI x9, x0, 9"
    assert at.session_state["seq"]["completed"] == 0


def test_failed_reassembly_keeps_old_program_but_says_so():
    at = assemble(new_app())
    assemble(at, "BAD x1")
    assert any("still loaded" in c.value for c in at.caption)
    assert len(at.session_state["program"]) == 12


def test_sequential_metrics_update_on_the_same_click():
    # Regression: the metric row used to lag one click behind the tables.
    at = assemble(new_app())
    assert metrics(at, SEQ)["PC"] == "0"
    at.button(key="seq_step").click().run()
    assert metrics(at, SEQ) == {
        "PC": "4", "Instructions completed": "1", "Status": "Running"}
    at.button(key="seq_step").click().run()
    assert metrics(at, SEQ)["PC"] == "8"
    at.button(key="seq_run").click().run()
    m = metrics(at, SEQ)
    assert m["Status"] == "Halted" and m["Instructions completed"] == "11"
    assert any("Program finished" in v for v in texts(at.tabs[SEQ].success))


def test_sequential_step_after_halt_explains_and_reset_works():
    at = assemble(new_app(), "ADDI x1, x0, 1")
    at.button(key="seq_step").click().run()
    assert metrics(at, SEQ)["Status"] == "Halted"
    at.button(key="seq_step").click().run()
    assert any("already halted" in v for v in texts(at.tabs[SEQ].info))
    assert metrics(at, SEQ)["Instructions completed"] == "1"          # no duplicate
    at.button(key="seq_reset").click().run()
    assert metrics(at, SEQ) == {
        "PC": "0", "Instructions completed": "0", "Status": "Ready"}


def test_sequential_fault_is_reported():
    at = assemble(new_app(), "ADDI x1, x0, 2\nLW x2, 0(x1)")
    at.button(key="seq_run").click().run()
    assert metrics(at, SEQ)["Status"] == "Faulted"
    assert any("unaligned" in v for v in texts(at.tabs[SEQ].error))
    at.button(key="seq_step").click().run()                  # persistent after rerun
    assert any("unaligned" in v for v in texts(at.tabs[SEQ].error))


def test_sequential_infinite_loop_warning():
    at = assemble(new_app(), "loop: BEQ x0, x0, loop")
    at.button(key="seq_run").click().run()
    assert any("1000 instructions" in v for v in texts(at.tabs[SEQ].warning))


def test_pipeline_metrics_follow_each_click_and_cpi_is_na_until_first_completion():
    at = assemble(new_app())
    at.button(key="pipe_step").click().run()
    m = metrics(at, PIPE)
    assert m["Clock cycles"] == "1" and m["Status"] == "Running"
    assert m["CPI"] == "n/a"
    for _ in range(4):
        at.button(key="pipe_step").click().run()
    m = metrics(at, PIPE)
    assert m["Clock cycles"] == "5" and m["Instructions completed"] == "1"
    assert m["CPI"] == "5.000"


def test_pipeline_run_to_completion_and_extra_step_message():
    at = assemble(new_app())
    at.button(key="pipe_run").click().run()
    m = metrics(at, PIPE)
    assert m["Status"] == "Completed" and m["Clock cycles"] == "18"
    assert (m["Instructions completed"], m["Stall cycles"], m["Branch flushes"]) == \
        ("11", "1", "1")
    assert m["Wrong-path instructions flushed"] == "1"
    at.button(key="pipe_step").click().run()
    assert any("already completed" in v for v in texts(at.tabs[PIPE].info))
    assert metrics(at, PIPE)["Clock cycles"] == "18"                 # unchanged
    at.button(key="pipe_reset").click().run()
    assert metrics(at, PIPE)["Clock cycles"] == "0"
    assert any("No cycles have run yet" in v for v in texts(at.tabs[PIPE].info))


def test_pipeline_cycle_inspector_reports_real_stall_and_flush_events():
    at = assemble(new_app())
    at.button(key="pipe_run").click().run()
    slider = at.select_slider(key="pipeline_cycle_inspector")
    timeline = at.session_state["pipe_cpu"]["timeline"]
    stall_cycle = next(r["Cycle"] for r in timeline if r["IF"] == "HELD")
    flush_cycle = next(r["Cycle"] for r in timeline if r["IF"] == "FLUSH")
    for cycle, label, other in ((stall_cycle, "Stall in this cycle", "Branch flush in this cycle"),
                                (flush_cycle, "Branch flush in this cycle", "Stall in this cycle")):
        slider.set_value(cycle).run()
        m = metrics(at, PIPE)
        assert m["Selected cycle"] == str(cycle)
        assert m[label] == "Yes" and m[other] == "No"
    events = [i.value for i in at.tabs[PIPE].info if "events:" in i.value]
    assert events and "Taken branch" in events[-1]


def test_pipeline_fault_is_reported_and_halts():
    at = assemble(new_app(), "ADDI x1, x0, 2\nLW x2, 0(x1)\nADDI x3, x0, 1")
    at.button(key="pipe_run").click().run()
    assert metrics(at, PIPE)["Status"] == "Faulted"
    assert any("unaligned" in v for v in texts(at.tabs[PIPE].error))
    cycles = metrics(at, PIPE)["Clock cycles"]
    at.button(key="pipe_step").click().run()
    assert metrics(at, PIPE)["Clock cycles"] == cycles


def test_performance_tab_metrics_are_consistent_with_the_engines():
    at = assemble(new_app())
    next(b for b in at.button if b.label == "Run Comparison").click().run()
    assert not at.exception
    assert any("Correctness check passed" in v for v in texts(at.tabs[PERF].success))
    m = metrics(at, PERF)
    assert m["Assembled instructions"] == "12"
    assert m["Instructions completed"] == "11"
    assert m["Never executed"] == "1"
    assert m["Pipeline cycles"] == "18"
    assert m["CPI"] == f"{18 / 11:.3f}" and m["IPC"] == f"{11 / 18:.3f}"
    assert m["Baseline cycles (model)"] == "55"
    assert m["Baseline CPI (model)"] == "5.000"
    assert m["Speedup vs. baseline (model)"] == f"{55 / 18:.2f}×"
    assert m["Cycle reduction"] == f"{(55 - 18) / 55 * 100:.1f}%"
    assert (m["Stall cycles"], m["Branch flushes"]) == ("1", "1")


def test_performance_tab_reports_faults_instead_of_numbers():
    at = assemble(new_app(), "ADDI x1, x0, 2\nLW x2, 0(x1)")
    next(b for b in at.button if b.label == "Run Comparison").click().run()
    assert any("Performance data is unavailable" in v for v in texts(at.tabs[PERF].error))


def test_machine_code_tab_encode_and_decode():
    at = new_app()
    next(b for b in at.button if b.label == "Encode Instruction").click().run()
    assert "Machine code: 0x002081B3" in texts(at.tabs[MACHINE].success)
    next(b for b in at.button if b.label == "Decode Machine Word").click().run()
    assert "Assembly: ADD x3, x1, x2" in texts(at.tabs[MACHINE].success)


def test_machine_code_tab_accepts_negative_branch_offset_and_rejects_garbage():
    at = new_app()
    at.text_input[0].set_value("BEQ x1, x2, -8").run()
    next(b for b in at.button if b.label == "Encode Instruction").click().run()
    assert not at.tabs[MACHINE].error
    assert any(v.startswith("Machine code: 0x") for v in texts(at.tabs[MACHINE].success))
    for bad in ("zzz", "0x1FFFFFFFF", "-1", "0x00000000"):
        at.text_input[1].set_value(bad).run()
        next(b for b in at.button if b.label == "Decode Machine Word").click().run()
        assert at.tabs[MACHINE].error, bad
        assert not at.exception


def test_automated_tests_tab_runs_and_results_persist():
    at = new_app()
    assert any("Click" in v for v in texts(at.tabs[TESTS].info))
    next(b for b in at.button if b.label == "Run All Automated Tests").click().run()
    m = metrics(at, TESTS)
    assert m["Failed"] == "0" and m["Passed"] == m["Total Tests"]
    assert int(m["Total Tests"]) >= 13
    assert any("All automated tests passed" in v for v in texts(at.tabs[TESTS].success))
    # results are still there after an unrelated interaction
    at.text_area(key="assembly_editor").set_value("ADDI x1, x0, 1").run()
    assert metrics(at, TESTS)["Passed"] == m["Passed"]
