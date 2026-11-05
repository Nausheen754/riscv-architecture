import streamlit as st
import pandas as pd

from riscv_engine import (
    MASK32, DEFAULT_PROGRAM,
    clean_source, decode_word,
    make_cpu, sequential_step, pipe_cycle, run_pipeline,
    calculate_comparison, safe_ratio, disassemble, run_bench,
)

# ============================================================
# RISC-V PROCESSOR SIMULATOR - Streamlit user interface
# The CPU engine lives in riscv_engine.py.
# Supported instructions: ADD, SUB, AND, OR, ADDI, LW, SW, BEQ
# ============================================================

st.set_page_config(
    page_title="RISC-V Processor Simulator",
    page_icon="⚙️",
    layout="wide"
)


# ---------------------- STREAMLIT UI -------------------------

st.title("⚙️ RISC-V Processor Simulator")
st.caption(
    "RV32I-inspired educational simulator | Sequential execution | "
    "Five-stage pipeline | Forwarding and hazard handling"
)

for key, default in (
    ("assembly_editor", DEFAULT_PROGRAM),   # text shown in the editor
    ("assembled_source", None),             # exact text that was last assembled
    ("program", []),
    ("seq", None),
    ("pipe_cpu", None),
    ("error", ""),
    ("comparison", None),
    ("comparison_error", ""),
    ("test_rows", None),
):
    if key not in st.session_state:
        st.session_state[key] = default


def program_is_stale():
    """True when the editor text differs from the program that was assembled."""
    return (
        bool(st.session_state.program)
        and st.session_state.assembled_source != st.session_state.assembly_editor
    )


def show_stale_warning():
    if program_is_stale():
        st.warning(
            "The assembly editor has changed since the last time you assembled. "
            "Everything on this tab still belongs to the **previously assembled** "
            "program. Click **Assemble Program** in the Assembly Editor tab to "
            "update it."
        )


def clear_execution_state():
    st.session_state.program = []
    st.session_state.assembled_source = None
    st.session_state.seq = None
    st.session_state.pipe_cpu = None
    st.session_state.comparison = None
    st.session_state.comparison_error = ""
    st.session_state.error = ""


def cpu_state_label(cpu, finished_label):
    if cpu["fault"]:
        return "Faulted"
    if cpu["halted"]:
        return finished_label
    return "Running" if cpu["completed"] or cpu["cycles"] else "Ready"


def pretty_ratio(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


with st.sidebar:
    st.header("Program")
    if st.button("Load Default Example", use_container_width=True):
        # The text area is controlled by its key, so set the key itself.
        st.session_state.assembly_editor = DEFAULT_PROGRAM
        clear_execution_state()

    st.caption("Supported: ADD, SUB, AND, OR, ADDI, LW, SW, BEQ")
    st.caption("Registers: x0–x31 | 32-bit words | byte addresses")
    if st.session_state.program:
        st.caption(
            f"Assembled: {len(st.session_state.program)} instructions"
            + (" (editor changed - reassemble)" if program_is_stale() else "")
        )
    else:
        st.caption("No program assembled yet.")

editor_tab, seq_tab, pipe_tab, machine_tab, performance_tab, tests_tab, help_tab = st.tabs([
    "Assembly Editor", "Sequential CPU", "Pipelined CPU",
    "Machine Code", "Performance Comparison", "Automated Tests", "Help"
])

with editor_tab:
    st.subheader("Assembly Program")
    st.text_area(
        "Edit your assembly program",
        height=300,
        key="assembly_editor",
        help="One instruction per line. Labels (name:) can be used with BEQ. "
             "Comments start with # or //.",
    )

    if st.button("Assemble Program", type="primary"):
        try:
            assembled = clean_source(st.session_state.assembly_editor)
            if not assembled:
                raise ValueError(
                    "Program is empty. Type at least one instruction, or click "
                    "'Load Default Example' in the sidebar."
                )
            st.session_state.program = assembled
            st.session_state.assembled_source = st.session_state.assembly_editor
            st.session_state.seq = make_cpu(assembled)
            st.session_state.pipe_cpu = make_cpu(assembled)
            st.session_state.comparison = None
            st.session_state.comparison_error = ""
            st.session_state.error = ""
            st.success(f"Successfully assembled {len(assembled)} instructions.")
        except ValueError as exc:
            st.session_state.error = str(exc)

    if st.session_state.error:
        st.error(st.session_state.error)
        if st.session_state.program:
            st.caption(
                "The previously assembled program is still loaded in the other tabs."
            )
    show_stale_warning()

    if st.session_state.program:
        st.markdown("#### Assembled program")
        asm_rows = []
        for i in st.session_state.program:
            asm_rows.append({
                "Address": i["pc"],
                "Assembly": i["text"],
                "Format": (
                    "R" if i["op"] in ("ADD", "SUB", "AND", "OR")
                    else "S" if i["op"] == "SW"
                    else "B" if i["op"] == "BEQ"
                    else "I"
                ),
                "Machine code": f"0x{i['word']:08X}",
                "Binary": f"{i['word']:032b}",
            })
        st.dataframe(pd.DataFrame(asm_rows), use_container_width=True, hide_index=True)

with seq_tab:
    st.subheader("Sequential CPU")
    show_stale_warning()
    if st.session_state.seq is None:
        st.info("Assemble a program first (Assembly Editor tab).")
    else:
        cpu = st.session_state.seq
        # The metric row is drawn AFTER the buttons are handled, so it always
        # shows the state produced by the click that triggered this run.
        status_box = st.container()
        notice = None

        b1, b2, b3 = st.columns(3)
        if b1.button("Step Instruction", key="seq_step"):
            if cpu["halted"]:
                notice = ("info", "The CPU has already halted. Click "
                          "'Reset Sequential CPU' to run the program again.")
            else:
                sequential_step(cpu)
        if b2.button("Run All", key="seq_run"):
            if cpu["halted"]:
                notice = ("info", "The CPU has already halted. Click "
                          "'Reset Sequential CPU' to run the program again.")
            else:
                count = 0
                while not cpu["halted"] and count < 1000:
                    sequential_step(cpu)
                    count += 1
                if not cpu["halted"]:
                    notice = ("warning", "Stopped after 1000 instructions - the "
                              "program may contain an infinite loop.")
        if b3.button("Reset Sequential CPU", key="seq_reset"):
            st.session_state.seq = make_cpu(st.session_state.program)
            st.rerun()

        with status_box:
            c1, c2, c3 = st.columns(3)
            c1.metric("PC", cpu["pc"])
            c2.metric("Instructions completed", cpu["completed"])
            c3.metric("Status", cpu_state_label(cpu, "Halted"))
        if cpu["fault"]:
            st.error(f"Execution stopped: {cpu['fault']}")
        elif cpu["halted"]:
            st.success("Program finished: the PC moved past the last instruction.")
        if notice:
            getattr(st, notice[0])(notice[1])

        st.markdown("#### Register File")
        st.dataframe(pd.DataFrame([
            {"Register": f"x{i}", "Decimal": v, "Hex": f"0x{v & MASK32:08X}"}
            for i, v in enumerate(cpu["regs"])
        ]), use_container_width=True, hide_index=True)

        st.markdown("#### Data Memory")
        st.dataframe(pd.DataFrame([
            {"Address": a, "Hex Address": f"0x{a:08X}",
             "Value": v, "Hex Value": f"0x{v:08X}"}
            for a, v in sorted(cpu["mem"].items())
        ]), use_container_width=True, hide_index=True)

        st.markdown("#### Execution History")
        if cpu["history"]:
            st.dataframe(pd.DataFrame(cpu["history"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No instructions executed yet.")

with pipe_tab:
    st.subheader("Real Five-Stage Pipeline")
    st.write(
        "Each Step Cycle advances the pipeline registers by one clock cycle. "
        "Forwarding, load-use stalls, and taken-branch flushing are handled by "
        "the cycle engine; branches are resolved in the EX stage."
    )
    show_stale_warning()

    if st.session_state.pipe_cpu is None:
        st.info("Assemble a program first (Assembly Editor tab).")
    else:
        cpu = st.session_state.pipe_cpu
        # Metrics are drawn after the buttons are handled (see Sequential tab).
        status_box = st.container()
        notice = None

        b1, b2, b3 = st.columns(3)
        if b1.button("Step Cycle", key="pipe_step"):
            if cpu["halted"]:
                notice = ("info", "The pipeline has already completed. Click "
                          "'Reset Pipeline' to run the program again.")
            else:
                pipe_cycle(cpu)
        if b2.button("Run Pipeline", key="pipe_run"):
            if cpu["halted"]:
                notice = ("info", "The pipeline has already completed. Click "
                          "'Reset Pipeline' to run the program again.")
            else:
                try:
                    run_pipeline(cpu)
                except RuntimeError as exc:
                    # A fault is also stored in cpu["fault"] and shown below.
                    if not cpu["fault"]:
                        notice = ("warning", str(exc))
        if b3.button("Reset Pipeline", key="pipe_reset"):
            st.session_state.pipe_cpu = make_cpu(st.session_state.program)
            st.rerun()

        with status_box:
            c0, c1, c2, c3, c4 = st.columns(5)
            c0.metric("Status", cpu_state_label(cpu, "Completed"))
            c1.metric("Clock cycles", cpu["cycles"])
            c2.metric("Instructions completed", cpu["completed"],
                      help="Instructions that reached the WB stage.")
            c3.metric("Stall cycles", cpu["stalls"],
                      help="Cycles lost to load-use hazards (one bubble each).")
            c4.metric("Branch flushes", cpu["flushes"],
                      help="Taken branches that redirected the PC. Wrong-path "
                           "instructions flushed: " + str(cpu["discarded"]))
        if cpu["fault"]:
            st.error(f"Pipeline stopped: {cpu['fault']}")
        elif cpu["halted"]:
            st.success("Pipeline finished: every instruction has left the WB stage.")
        if notice:
            getattr(st, notice[0])(notice[1])

        st.markdown("#### Current Pipeline Registers")
        pipe_rows = []
        for name, value in cpu["pipe"].items():
            pipe_rows.append({
                "Pipeline register": name,
                "Contents": value["inst"]["text"] if value else "BUBBLE / EMPTY"
            })
        st.dataframe(pd.DataFrame(pipe_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Visual Pipeline Diagram")
        st.caption(
            "Each column is a clock cycle and each row is a pipeline stage. "
            "The colors distinguish stages; STALL and FLUSH cells are highlighted."
        )
        if cpu["timeline"]:
            timeline_df = pd.DataFrame(cpu["timeline"]).sort_values("Cycle")
            stage_names = ["IF", "ID", "EX", "MEM", "WB"]
            cycle_options = [int(c) for c in timeline_df["Cycle"].tolist()]
            selected_cycle = st.select_slider(
                "Inspect a clock cycle",
                options=cycle_options,
                value=cycle_options[-1],
                format_func=lambda c: f"Cycle {c}",
                key="pipeline_cycle_inspector",
            )

            # Keep the visual grid compact: show a moving window around the
            # selected cycle. The selector still allows inspection of every cycle.
            visible_cycle_count = 8
            total_cycles = len(cycle_options)
            selected_index = cycle_options.index(selected_cycle)
            window_start = max(0, selected_index - visible_cycle_count // 2 + 1)
            window_start = min(window_start, max(0, total_cycles - visible_cycle_count))
            visible_cycles = cycle_options[window_start:window_start + visible_cycle_count]

            diagram_rows = []
            for stage in stage_names:
                stage_row = {"Stage": stage}
                for cycle_number in visible_cycles:
                    cycle_row = timeline_df[timeline_df["Cycle"] == cycle_number].iloc[0]
                    value = str(cycle_row.get(stage, "-"))
                    stage_row[f"C{cycle_number}"] = value if value != "-" else "·"
                diagram_rows.append(stage_row)
            diagram_df = pd.DataFrame(diagram_rows)
            st.caption(
                f"Showing cycles {visible_cycles[0]}–{visible_cycles[-1]} of {total_cycles}. "
                "Use the selector above to move the highlighted cycle through the full timeline."
            )
            selected_row = timeline_df[timeline_df["Cycle"] == selected_cycle].iloc[0]
            detail_cols = st.columns(4)
            detail_cols[0].metric("Selected cycle", selected_cycle)
            detail_cols[1].metric("Stall in this cycle", "Yes" if selected_row["IF"] == "HELD" else "No")
            detail_cols[2].metric("Branch flush in this cycle", "Yes" if selected_row["IF"] == "FLUSH" else "No")
            detail_cols[3].metric("WB instruction", str(selected_row.get("WB", "-")))
            selected_events = next(
                (e["Events"] for e in cpu["events"] if e["Cycle"] == selected_cycle),
                "No event recorded.",
            )
            st.info(f"**Cycle {selected_cycle} events:** {selected_events}")

            stage_colors = {
                "IF": "#DCEBFF", "ID": "#E7DFFF", "EX": "#DDF5E3",
                "MEM": "#FFF0CC", "WB": "#F9DDE8",
            }
            selected_col = f"C{selected_cycle}"
            def style_pipeline_row(row):
                stage = str(row.get("Stage", ""))
                styles = []
                for col in row.index:
                    value = str(row[col]).lower()
                    if col == "Stage":
                        styles.append("font-weight: bold; background-color: #E8EAED; color: #202124;")
                    elif "flush" in value:
                        styles.append("background-color: #F8B4B4; color: #7F1D1D; font-weight: bold; border: 2px solid #DC2626;")
                    elif "held" in value or "stall" in value or "bubble" in value:
                        styles.append("background-color: #FFE0B2; color: #7C2D12; font-weight: bold;")
                    elif value == "·" or value == "-":
                        styles.append("background-color: #F3F4F6; color: #9CA3AF;")
                    else:
                        styles.append(f"background-color: {stage_colors.get(stage, '#FFFFFF')}; color: #202124;")
                    if col == selected_col:
                        styles[-1] += " border-left: 4px solid #1D4ED8; border-right: 4px solid #1D4ED8; border-top: 3px solid #1D4ED8; border-bottom: 3px solid #1D4ED8; font-weight: 700;"
                return styles

            styled_diagram = diagram_df.style.apply(style_pipeline_row, axis=1)
            st.dataframe(
                styled_diagram,
                use_container_width=True,
                hide_index=True,
                height=270,
            )
            st.markdown(
                "**Legend:** 🔵 IF = Fetch · 🟣 ID = Decode · 🟢 EX = Execute · "
                "🟡 MEM = Memory · 🩷 WB = Write Back · 🟠 HELD (stall) · 🔴 FLUSH · "
                "· = empty stage (pipeline fill/drain or bubble)"
            )
            st.markdown("#### Selected Cycle Details")
            st.dataframe(pd.DataFrame([selected_row]), use_container_width=True, hide_index=True)
            st.markdown("#### Detailed Cycle Timeline")
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        else:
            st.info("No cycles have run yet. Click Step Cycle or Run Pipeline to "
                    "generate the visual pipeline diagram.")

        st.markdown("#### Hazard and Forwarding Events")
        if cpu["events"]:
            st.dataframe(pd.DataFrame(cpu["events"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No cycles have run yet.")

        # CPI = clock cycles so far / instructions completed so far.
        cpi = safe_ratio(cpu["cycles"], cpu["completed"])
        ipc = safe_ratio(cpu["completed"], cpu["cycles"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CPI", pretty_ratio(cpi),
                  help="Clock cycles / instructions completed. Shows n/a until "
                       "the first instruction completes. Includes pipeline "
                       "fill, stalls and flushes.")
        m2.metric("IPC", pretty_ratio(ipc), help="Instructions completed / clock cycles.")
        m3.metric("Forwarded operands", cpu["forwarding"],
                  help="Operand values taken from EX/MEM or MEM/WB instead of "
                       "the register file. This counts operands, not cycles, "
                       "and forwarding never costs a cycle.")
        m4.metric("Wrong-path instructions flushed", cpu["discarded"],
                  help="Instructions fetched after a taken branch and discarded "
                       "before they could complete.")

        st.markdown("#### Pipeline Register File")
        st.dataframe(pd.DataFrame([
            {"Register": f"x{i}", "Decimal": v, "Hex": f"0x{v & MASK32:08X}"}
            for i, v in enumerate(cpu["regs"])
        ]), use_container_width=True, hide_index=True)

        st.markdown("#### Pipeline Data Memory")
        st.dataframe(pd.DataFrame([
            {"Address": a, "Value": v, "Hex": f"0x{v:08X}"}
            for a, v in sorted(cpu["mem"].items())
        ]), use_container_width=True, hide_index=True)

with machine_tab:
    st.subheader("Machine-Code Encoder / Decoder")
    st.write(
        "**Assembly** is the human-readable text you type (for example "
        "`ADD x3, x1, x2`). **Machine code** is the 32-bit binary word the "
        "processor would actually fetch. This tab converts between the two for "
        "the supported instructions. (The CPU tabs execute the parsed assembly "
        "instruction; they do not decode these words.)"
    )
    enc_col, dec_col = st.columns(2)

    with enc_col:
        st.markdown("#### Assembly → machine code")
        encode_text = st.text_input("Assembly instruction", "ADD x3, x1, x2")
        if st.button("Encode Instruction"):
            try:
                # check_targets=False: a lone BEQ may use a negative offset.
                program = clean_source(encode_text, check_targets=False)
                if len(program) != 1:
                    raise ValueError("Enter exactly one instruction.")
                word = program[0]["word"]
                st.success(f"Machine code: 0x{word:08X}")
                st.code(f"Binary: {word:032b}\nHex: 0x{word:08X}")
                st.json(decode_word(word))
            except ValueError as exc:
                st.error(str(exc))

    with dec_col:
        st.markdown("#### Machine code → assembly")
        hex_text = st.text_input("Machine word (hexadecimal)", "0x002081B3")
        if st.button("Decode Machine Word"):
            try:
                try:
                    word = int(hex_text.strip(), 16)
                except ValueError:
                    raise ValueError(
                        f"'{hex_text}' is not a hexadecimal number. "
                        "Example: 0x002081B3"
                    )
                if not 0 <= word <= MASK32:
                    raise ValueError("Enter a 32-bit word (0x00000000 to 0xFFFFFFFF).")
                decoded = decode_word(word)
                st.success(f"Assembly: {disassemble(decoded)}")
                st.code(f"Binary: {word:032b}\nHex: 0x{word:08X}")
                st.json(decoded)
            except ValueError as exc:
                st.error(str(exc))

with performance_tab:
    st.subheader("Sequential vs. Pipelined Performance")
    st.write(
        "This comparison runs fresh copies of the currently assembled program, "
        "so it does not depend on how far you have manually stepped either CPU. "
        "Pipeline numbers are **measured** from the simulator's cycle-by-cycle "
        "trace. The serial five-stage figure is a **model** (see below), not a "
        "measurement."
    )
    show_stale_warning()

    if not st.session_state.program:
        st.info("Assemble a program first (Assembly Editor tab).")
    else:
        if st.button("Run Comparison", type="primary", key="run_comparison"):
            try:
                st.session_state.comparison = calculate_comparison(
                    st.session_state.program
                )
                st.session_state.comparison_error = ""
            except RuntimeError as exc:
                st.session_state.comparison = None
                st.session_state.comparison_error = (
                    f"Performance data is unavailable. {exc}"
                )

        if st.session_state.get("comparison_error"):
            st.error(st.session_state.comparison_error)

        comparison = st.session_state.get("comparison")
        if comparison:
            seq_cpu = comparison["seq"]
            pipe_cpu = comparison["pipe"]
            mismatches = comparison["mismatches"]

            if mismatches:
                st.error(
                    f"Correctness check failed: {len(mismatches)} difference(s) "
                    "between the sequential and pipelined final state."
                )
                st.dataframe(
                    pd.DataFrame(mismatches),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "Correctness check passed: final registers, data memory, PC "
                    "and completed-instruction count match."
                )

            st.markdown("#### Instruction counts")
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Assembled instructions", comparison["assembled_count"],
                      help="Static length of the program.")
            i2.metric("Instructions completed", comparison["instruction_count"],
                      help="Instructions that reached WB. Loops count every "
                           "repetition; skipped code counts zero.")
            i3.metric("Never executed", comparison["never_executed"],
                      help="Assembled instructions that were skipped by branches.")
            i4.metric("Wrong-path instructions flushed", comparison["discarded_count"],
                      help="Fetched after a taken branch, then discarded before "
                           "completing.")

            st.markdown("#### Pipelined CPU (measured)")
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Pipeline cycles", pipe_cpu["cycles"],
                      help="Clock cycles from the cycle trace: fill + execution "
                           "+ stalls + flush bubbles + drain.")
            p2.metric("CPI", pretty_ratio(comparison["pipe_cpi"]),
                      help="Pipeline cycles / instructions completed.")
            p3.metric("IPC", pretty_ratio(comparison["pipe_ipc"]),
                      help="Instructions completed / pipeline cycles.")
            p4.metric("Stall cycles", pipe_cpu["stalls"],
                      help="Load-use bubbles. Each stall costs one cycle.")
            p5.metric("Branch flushes", pipe_cpu["flushes"],
                      help="Taken branches. Each redirects the PC from EX.")
            p6.metric("Forwarded operands", pipe_cpu["forwarding"],
                      help="Operand values bypassed from EX/MEM or MEM/WB. "
                           "Counts operands (not cycles); costs no cycles.")

            st.markdown("#### Serial five-stage baseline (model)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Baseline cycles (model)", comparison["serial_five_stage_cycles"],
                      help="Model: completed instructions x 5. A processor that "
                           "finishes one instruction's five stages before "
                           "starting the next.")
            m2.metric("Baseline CPI (model)", "5.000" if comparison["instruction_count"] else "n/a",
                      help="By definition of the model: 5 cycles per instruction.")
            m3.metric(
                "Speedup vs. baseline (model)",
                "n/a" if comparison["speedup"] is None else f'{comparison["speedup"]:.2f}×',
                help="Baseline cycles / pipeline cycles. A teaching estimate, "
                     "not a measured hardware benchmark.",
            )
            m4.metric("Cycle reduction", f'{comparison["cycle_reduction_pct"]:.1f}%',
                      help="(Baseline cycles - pipeline cycles) / baseline cycles.")
            st.caption(
                f"The Sequential CPU tab executes one instruction per software "
                f"step ({seq_cpu['cycles']} steps for this program). A step is "
                "not a clock cycle, so it is deliberately not compared with "
                "pipeline cycles."
            )

            st.markdown("#### What the performance result means")
            st.info(comparison["performance_verdict"])
            st.caption(
                "A speedup above 1.00× means the pipeline used fewer cycles than "
                "the model baseline; below 1.00× means more. The baseline assumes "
                "five serial stage-cycles per completed instruction and does not "
                "represent physical processor clock time."
            )

            st.markdown("#### Where the pipeline cycles went")
            flush_bubbles = (
                comparison["extra_cycles"] - pipe_cpu["stalls"]
            )
            breakdown = pd.DataFrame({
                "Component": [
                    "One cycle per completed instruction",
                    "Pipeline fill/drain (4 cycles)",
                    "Load-use stall cycles (counted)",
                    "Branch-flush bubbles (derived)",
                    "Total pipeline cycles",
                ],
                "Cycles": [
                    comparison["instruction_count"],
                    4 if comparison["instruction_count"] else 0,
                    pipe_cpu["stalls"],
                    flush_bubbles,
                    pipe_cpu["cycles"],
                ],
            })
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
            st.caption(
                "Flush bubbles are derived as total cycles minus the other "
                "components. Each taken branch costs up to 2 bubbles, and fewer "
                "when there is nothing left to fetch."
            )

            st.markdown("#### Cycle-count comparison")
            chart_df = pd.DataFrame({
                "Execution model": [
                    "Serial 5-stage baseline (model)",
                    "Pipelined (measured)",
                ],
                "Cycles": [
                    comparison["serial_five_stage_cycles"],
                    pipe_cpu["cycles"],
                ],
            }).set_index("Execution model")
            st.bar_chart(chart_df)

            st.markdown("#### Hazard events (counts, not cycles)")
            event_df = pd.DataFrame({
                "Event": [
                    "Stall cycles",
                    "Branch flushes",
                    "Wrong-path instructions flushed",
                    "Forwarded operands",
                ],
                "Count": [
                    pipe_cpu["stalls"],
                    pipe_cpu["flushes"],
                    pipe_cpu["discarded"],
                    pipe_cpu["forwarding"],
                ],
            }).set_index("Event")
            st.bar_chart(event_df)

            st.markdown("#### Final register comparison")
            register_rows = []
            for i in range(32):
                sv = seq_cpu["regs"][i]
                pv = pipe_cpu["regs"][i]
                register_rows.append({
                    "Register": f"x{i}",
                    "Sequential": sv,
                    "Pipelined": pv,
                    "Match": sv == pv,
                    "Hex": f"0x{(pv & MASK32):08X}",
                })
            st.dataframe(
                pd.DataFrame(register_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Final memory comparison")
            memory_addresses = sorted(
                set(seq_cpu["mem"]) | set(pipe_cpu["mem"])
            )
            if memory_addresses:
                memory_rows = []
                for address in memory_addresses:
                    sv = seq_cpu["mem"].get(address, 0)
                    pv = pipe_cpu["mem"].get(address, 0)
                    memory_rows.append({
                        "Address": address,
                        "Sequential": sv,
                        "Pipelined": pv,
                        "Match": sv == pv,
                    })
                st.dataframe(
                    pd.DataFrame(memory_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("The program did not write to data memory.")

            st.warning(
                "Performance figures are meaningful only after the correctness check "
                "passes. The five-stage serial baseline is an illustrative comparison "
                "model, not a measurement of a real processor."
            )
        else:
            st.caption("Click Run Comparison to execute both CPUs from a clean state.")

with help_tab:
    st.subheader("How the simulator works")
    st.markdown("""
    **IF — Instruction Fetch:** fetches the instruction at the PC.

    **ID — Instruction Decode:** identifies the instruction and reads registers.

    **EX — Execute:** performs the ALU operation, computes a memory address, or
    compares branch operands (branches are resolved here).

    **MEM — Memory:** performs a load or store.

    **WB — Write Back:** writes the result to the destination register.

    **Forwarding:** supplies a recent result from EX/MEM or MEM/WB to an
    instruction in EX without waiting for write-back. The newest result always
    wins. A load result is not available from EX/MEM, so a dependent
    instruction directly after a load must stall.

    **Load-use stall:** one bubble is inserted when the instruction after a
    load needs the loaded register (including as a store address/data or branch
    operand).

    **Branch flush:** branches are predicted not-taken. A taken branch is found
    in EX, the wrong-path instruction in ID is discarded, the fetch in that
    cycle is cancelled, and the PC is redirected. The penalty is up to 2 cycles.

    **Metric definitions:** *Instructions completed* = instructions that
    reached WB. *Pipeline cycles* = clock cycles recorded by the cycle engine.
    *CPI* = cycles / instructions completed. *Forwarded operands* counts operand
    values bypassed (an `ADD x2, x1, x1` can forward two). The *serial
    five-stage baseline* is a model (completed instructions x 5 cycles), not a
    measurement.

    **Memory model:** byte addresses, 32-bit words. Loads and stores must use
    4-byte-aligned, non-negative addresses; otherwise the CPU halts with a
    fault message. Unwritten aligned words read as zero. Registers hold signed
    32-bit values (arithmetic wraps on overflow); memory stores the raw word.
    x0 always remains zero.

    **Assembly syntax:** commas are required between operands; memory operands
    are written `offset(base)` such as `8(x2)`; only BEQ accepts labels; BEQ
    offsets are relative to the BEQ and must be multiples of 4. Comments start
    with `#` or `//`. ADDI immediates are 12-bit signed.

    **Scope:** this is an educational subset (ADD, SUB, AND, OR, ADDI, LW, SW,
    BEQ), not a complete implementation of the RISC-V ISA. The machine-code tab
    encodes and decodes these instructions; the CPU modes execute the parsed
    assembly instructions.
    """)


# ---------------------- AUTOMATED REGRESSION TESTS ----------------------
with tests_tab:
    st.subheader("Automated Test Bench")
    st.caption(
        "Runs each program on both CPUs and checks registers, memory, completed "
        "instructions, exact stall/flush/cycle counts, machine-code round trips, "
        "assembler error handling and fault safety. The developer test suite "
        "(`pytest`) covers the same engine in more depth."
    )

    if st.button("Run All Automated Tests", type="primary", key="run_all_tests"):
        with st.spinner("Running tests..."):
            st.session_state.test_rows = run_bench()

    rows = st.session_state.test_rows
    if rows is None:
        st.info("Click **Run All Automated Tests** to execute the regression suite.")
    else:
        results_df = pd.DataFrame(rows)
        passed_count = int((results_df["Status"] == "PASS").sum())
        failed_count = len(results_df) - passed_count
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Tests", len(results_df))
        c2.metric("Passed", passed_count)
        c3.metric("Failed", failed_count)
        if failed_count:
            st.error(f"{failed_count} test(s) failed. Failure details are listed below.")
            for row in rows:
                if row["Status"] == "FAIL":
                    st.markdown(f"**❌ {row['Test']}**")
                    st.code(row["Details"], language=None)
        else:
            st.success("All automated tests passed.")
        st.markdown("### Test Results")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
