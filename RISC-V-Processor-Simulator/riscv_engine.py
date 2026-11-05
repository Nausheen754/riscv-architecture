"""RISC-V Processor Simulator - simulation engine (no Streamlit code here).

Contains the assembler, machine-code encoder/decoder, sequential CPU and the
five-stage pipelined CPU. app.py imports from this module for the user
interface, and tests/ imports it directly so the CPU can be tested without
starting Streamlit.

Supported instructions (educational RV32I subset): ADD, SUB, AND, OR, ADDI,
LW, SW, BEQ.
"""

import re

MASK32 = 0xFFFFFFFF
STAGES = ["IF", "ID", "EX", "MEM", "WB"]


# ------------------------- ASSEMBLER -------------------------

def clean_source(source, check_targets=True):
    """Assemble source text into a list of instruction dicts.

    Removes comments and resolves labels to byte offsets. With
    check_targets=False a BEQ may point before address 0 (used by the
    single-instruction machine-code encoder, which has no program around it).
    """
    lines = []
    labels = {}
    pc = 0

    for lineno, raw in enumerate(source.splitlines(), start=1):
        line = raw.split("#")[0].split("//")[0].strip()
        if not line:
            continue

        while ":" in line:
            label, rest = line.split(":", 1)
            label = label.strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", label):
                raise ValueError(f"Line {lineno}: invalid label '{label}'")
            if label in labels:
                raise ValueError(f"Line {lineno}: duplicate label '{label}'")
            labels[label] = pc
            line = rest.strip()
            if not line:
                break

        if line:
            lines.append((pc, line, lineno))
            pc += 4

    program = []
    for address, text, lineno in lines:
        pieces = text.split(None, 1)
        op = pieces[0].upper()
        rest = pieces[1] if len(pieces) > 1 else ""

        try:
            args = split_operands(rest)

            if op in ("ADD", "SUB", "AND", "OR"):
                if len(args) != 3:
                    raise ValueError("Expected rd, rs1, rs2")
                rd, rs1, rs2 = map(parse_reg, args)
                inst = dict(op=op, rd=rd, rs1=rs1, rs2=rs2, imm=0)

            elif op == "ADDI":
                if len(args) != 3:
                    raise ValueError("Expected rd, rs1, immediate")
                rd, rs1 = parse_reg(args[0]), parse_reg(args[1])
                imm = parse_imm(args[2])
                check_signed(imm, 12, "ADDI immediate")
                inst = dict(op=op, rd=rd, rs1=rs1, rs2=0, imm=imm)

            elif op in ("LW", "SW"):
                if len(args) != 2:
                    raise ValueError(
                        "Expected register, offset(base), for example "
                        f"{op} x1, 0(x2)"
                    )
                reg = parse_reg(args[0])
                imm, base = parse_mem_operand(args[1])
                check_signed(imm, 12, f"{op} offset")
                if imm % 4:
                    raise ValueError(f"{op} offset must be a multiple of 4")
                if op == "LW":
                    inst = dict(op=op, rd=reg, rs1=base, rs2=0, imm=imm)
                else:
                    inst = dict(op=op, rd=0, rs1=base, rs2=reg, imm=imm)

            elif op == "BEQ":
                if len(args) != 3:
                    raise ValueError("Expected rs1, rs2, offset or label")
                rs1, rs2 = parse_reg(args[0]), parse_reg(args[1])
                imm = parse_imm(args[2], address, labels)
                check_signed(imm, 13, "BEQ offset")
                if imm % 4:
                    raise ValueError(
                        "BEQ offset must be a multiple of 4 "
                        "(every instruction is 4 bytes)"
                    )
                if check_targets and address + imm < 0:
                    raise ValueError(
                        f"BEQ target address {address + imm} is before the "
                        "start of the program"
                    )
                inst = dict(op=op, rd=0, rs1=rs1, rs2=rs2, imm=imm)

            else:
                raise ValueError(f"Unsupported instruction: {op}")

            inst["pc"] = address
            inst["text"] = text
            inst["word"] = encode_instruction(inst)
            program.append(inst)

        except ValueError as exc:
            raise ValueError(
                f"Assembly error on line {lineno} (address {address}), "
                f"'{text}': {exc}"
            ) from exc

    return program


def split_operands(rest):
    """Split 'x1, x2, x3' into ['x1', 'x2', 'x3']; commas are required."""
    if not rest.strip():
        return []
    operands = [part.strip() for part in rest.split(",")]
    if any(part == "" for part in operands):
        raise ValueError("Empty operand - check for a missing or extra comma")
    return operands


def parse_reg(token):
    token = token.strip().lower()
    if not re.fullmatch(r"x\d+", token):
        raise ValueError(f"Invalid register '{token}'. Use x0 through x31.")
    reg = int(token[1:])
    if not 0 <= reg <= 31:
        raise ValueError(f"Register out of range: {token}")
    return reg


def parse_imm(token, pc=None, labels=None):
    """Parse a number (decimal, 0x hex, 0b binary).

    Labels are accepted only when `labels` is given (BEQ); they become a
    pc-relative byte offset.
    """
    token = token.strip()
    if labels is not None and token in labels:
        return labels[token] - pc
    if re.fullmatch(r"[+-]?(0[xX][0-9a-fA-F]+|0[bB][01]+|\d+)", token):
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token, 10)
        return int(token, 0)
    if labels is not None:
        raise ValueError(f"Invalid offset or unknown label: '{token}'")
    raise ValueError(
        f"Invalid immediate '{token}'. Use a decimal, 0x hex or 0b binary number."
    )


def parse_mem_operand(token):
    """Parse 'offset(base)' such as '8(x2)' into (offset, base register)."""
    match = re.fullmatch(r"(.*?)\s*\(\s*([^()\s]+)\s*\)", token.strip())
    if not match:
        raise ValueError(
            f"Memory operand must look like offset(base), for example "
            f"8(x2); got '{token}'"
        )
    offset_text, base_text = match.groups()
    offset = parse_imm(offset_text) if offset_text.strip() else 0
    return offset, parse_reg(base_text)


def check_signed(value, bits, name):
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError(f"{name} does not fit in {bits} signed bits: {value}")


# --------------------- MACHINE CODE -------------------------

def encode_instruction(i):
    op = i["op"]
    rd, rs1, rs2, imm = i["rd"], i["rs1"], i["rs2"], i["imm"]

    if op in ("ADD", "SUB", "AND", "OR"):
        funct3 = {"ADD": 0, "SUB": 0, "AND": 7, "OR": 6}[op]
        funct7 = 0x20 if op == "SUB" else 0
        return (
            (funct7 << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | 0x33
        )

    if op == "ADDI":
        return ((imm & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13

    if op == "LW":
        return ((imm & 0xFFF) << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x03

    if op == "SW":
        u = imm & 0xFFF
        return (
            ((u >> 5) << 25) | (rs2 << 20) | (rs1 << 15)
            | (2 << 12) | ((u & 0x1F) << 7) | 0x23
        )

    if op == "BEQ":
        u = imm & 0x1FFF
        return (
            (((u >> 12) & 1) << 31)
            | (((u >> 5) & 0x3F) << 25)
            | (rs2 << 20) | (rs1 << 15)
            | (((u >> 1) & 0xF) << 8)
            | (((u >> 11) & 1) << 7) | 0x63
        )

    raise ValueError(f"Cannot encode {op}")


def decode_word(word):
    """Decode a 32-bit machine word into a supported instruction."""
    word &= MASK32
    opcode = word & 0x7F
    rd = (word >> 7) & 31
    funct3 = (word >> 12) & 7
    rs1 = (word >> 15) & 31
    rs2 = (word >> 20) & 31
    funct7 = (word >> 25) & 127

    if opcode == 0x33:
        lookup = {(0, 0): "ADD", (0, 32): "SUB", (7, 0): "AND", (6, 0): "OR"}
        op = lookup.get((funct3, funct7))
        if op is None:
            raise ValueError("Unsupported R-type machine instruction")
        return dict(op=op, rd=rd, rs1=rs1, rs2=rs2, imm=0)

    if opcode in (0x13, 0x03):
        if opcode == 0x13 and funct3 == 0:
            op = "ADDI"
        elif opcode == 0x03 and funct3 == 2:
            op = "LW"
        else:
            raise ValueError("Unsupported I-type machine instruction")
        imm = sign_extend(word >> 20, 12)
        return dict(op=op, rd=rd, rs1=rs1, rs2=0, imm=imm)

    if opcode == 0x23 and funct3 == 2:
        imm = (((word >> 25) & 127) << 5) | ((word >> 7) & 31)
        return dict(op="SW", rd=0, rs1=rs1, rs2=rs2, imm=sign_extend(imm, 12))

    if opcode == 0x63 and funct3 == 0:
        imm = (
            (((word >> 31) & 1) << 12)
            | (((word >> 25) & 63) << 5)
            | (((word >> 8) & 15) << 1)
            | (((word >> 7) & 1) << 11)
        )
        return dict(op="BEQ", rd=0, rs1=rs1, rs2=rs2, imm=sign_extend(imm, 13))

    raise ValueError(f"Unsupported opcode: {opcode:#x}")


def disassemble(inst):
    """Turn a decoded/assembled instruction dict back into assembly text."""
    op, rd, rs1, rs2, imm = (inst[k] for k in ("op", "rd", "rs1", "rs2", "imm"))
    if op in ("ADD", "SUB", "AND", "OR"):
        return f"{op} x{rd}, x{rs1}, x{rs2}"
    if op == "ADDI":
        return f"ADDI x{rd}, x{rs1}, {imm}"
    if op == "LW":
        return f"LW x{rd}, {imm}(x{rs1})"
    if op == "SW":
        return f"SW x{rs2}, {imm}(x{rs1})"
    if op == "BEQ":
        return f"BEQ x{rs1}, x{rs2}, {imm}"
    raise ValueError(f"Cannot disassemble {op}")


def sign_extend(value, bits):
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


# ----------------------- CPU STATE ---------------------------

def make_cpu(program):
    return {
        "program": program,
        "regs": [0] * 32,
        "mem": {},
        "pc": 0,
        "halted": not program,      # an empty program has nothing to run
        "fault": None,
        "history": [],
        "cycles": 0,
        "completed": 0,
        "stalls": 0,
        "flushes": 0,
        "discarded": 0,
        "forwarding": 0,
        "pipe": {s: None for s in ["IFID", "IDEX", "EXMEM", "MEMWB"]},
        "timeline": [],
        "events": [],
        "next_uid": 1,
        "redirects": 0,
    }


def get_inst(program, pc):
    if pc < 0 or pc % 4 != 0:
        raise ValueError(f"Invalid instruction PC: {pc}")
    index = pc // 4
    return program[index] if index < len(program) else None


def to_signed32(value):
    """Wrap any integer into the signed 32-bit range used by the registers."""
    value &= MASK32
    return value - (1 << 32) if value >= (1 << 31) else value


def alu(op, a, b):
    if op in ("ADD", "ADDI", "LW", "SW"):
        value = a + b
    elif op == "SUB":
        value = a - b
    elif op == "AND":
        value = a & b
    elif op == "OR":
        value = a | b
    else:
        value = 0
    return to_signed32(value)


def read_mem(cpu, address):
    if address < 0 or address % 4 != 0:
        raise ValueError(f"Invalid or unaligned word load at address {address}")
    # Memory stores raw 32-bit words (0..2^32-1); registers hold signed
    # values, so a loaded word is converted back to signed.
    return to_signed32(cpu["mem"].get(address, 0))


def write_mem(cpu, address, value):
    if address < 0 or address % 4 != 0:
        raise ValueError(f"Invalid or unaligned word store at address {address}")
    cpu["mem"][address] = value & MASK32


# -------------------- SEQUENTIAL CPU -------------------------

def sequential_step(cpu):
    """Execute one instruction. Does nothing once the CPU has halted.

    A memory fault (unaligned or negative address) halts the CPU: the
    faulting instruction does not complete and cpu["fault"] explains why.
    """
    if cpu["halted"]:
        return
    inst = get_inst(cpu["program"], cpu["pc"])
    if inst is None:
        cpu["halted"] = True
        return

    op = inst["op"]
    regs = cpu["regs"]
    a, b = regs[inst["rs1"]], regs[inst["rs2"]]
    next_pc = cpu["pc"] + 4
    result = None

    try:
        if op in ("ADD", "SUB", "AND", "OR"):
            result = alu(op, a, b)
        elif op == "ADDI":
            result = alu(op, a, inst["imm"])
        elif op == "LW":
            result = read_mem(cpu, alu("ADD", a, inst["imm"]))
        elif op == "SW":
            write_mem(cpu, alu("ADD", a, inst["imm"]), b)
    except ValueError as exc:
        cpu["fault"] = (
            f"{exc} (instruction '{inst['text']}' at PC {cpu['pc']})"
        )
        cpu["halted"] = True
        return

    if op == "BEQ":
        if a == b:
            next_pc = cpu["pc"] + inst["imm"]

    if result is not None and inst["rd"] != 0:
        regs[inst["rd"]] = result
    regs[0] = 0

    cpu["history"].append({
        "PC": cpu["pc"], "Instruction": inst["text"],
        "Result": result, "Next PC": next_pc
    })
    cpu["pc"] = next_pc
    cpu["completed"] += 1
    cpu["cycles"] += 1
    cpu["halted"] = get_inst(cpu["program"], cpu["pc"]) is None


# ------------------- PIPELINE CPU ----------------------------
# Pipeline registers hold state between cycles.
# All stage computations use the old pipeline state.
# Branches resolve in EX. Taken branches flush IF/ID and ID/EX.
# Forwarding: EX/MEM (except loads) and MEM/WB to EX.
# Load-use dependency: one cycle held in ID with an EX bubble.

def source_regs(inst):
    if not inst:
        return set()
    op = inst["op"]
    if op in ("ADD", "SUB", "AND", "OR", "BEQ"):
        return {inst["rs1"], inst["rs2"]}
    if op in ("ADDI", "LW"):
        return {inst["rs1"]}
    if op == "SW":
        return {inst["rs1"], inst["rs2"]}
    return set()


def writes_rd(inst):
    return bool(inst and inst["op"] in ("ADD", "SUB", "AND", "OR", "ADDI", "LW")
                and inst["rd"] != 0)


def pipeline_fault(cpu, row, inst, message):
    """Stop the pipeline after a memory fault in the MEM stage.

    Older instructions have already committed in WB this cycle; the faulting
    instruction and everything younger are discarded without completing.
    """
    text = f"{message} (instruction '{inst['text']}' at PC {inst['pc']})"
    row["MEM"] = f"FAULT: {inst['text']}"
    cpu["cycles"] += 1
    cpu["fault"] = text
    cpu["halted"] = True
    cpu["pipe"] = {k: None for k in cpu["pipe"]}
    cpu["timeline"].append(row)
    cpu["events"].append({"Cycle": cpu["cycles"], "Events": f"Memory fault: {text}"})


def pipe_cycle(cpu):
    """Advance the pipeline by one clock cycle. Does nothing once halted."""
    if cpu["halted"]:
        return

    p = cpu["pipe"]
    old_ifid = p["IFID"]
    old_idex = p["IDEX"]
    old_exmem = p["EXMEM"]
    old_memwb = p["MEMWB"]
    regs = cpu["regs"]

    new_ifid = old_ifid
    new_idex = None
    new_exmem = None
    new_memwb = None

    events = []
    row = {"Cycle": cpu["cycles"] + 1}
    for s in STAGES:
        row[s] = "-"

    # WB: commit previous cycle's MEM/WB value.
    if old_memwb:
        inst = old_memwb["inst"]
        row["WB"] = inst["text"]
        if writes_rd(inst):
            regs[inst["rd"]] = old_memwb["value"]
        regs[0] = 0
        cpu["completed"] += 1

    # MEM: load/store using EX/MEM.
    if old_exmem:
        inst = old_exmem["inst"]
        row["MEM"] = inst["text"]
        value = old_exmem["alu"]
        try:
            if inst["op"] == "LW":
                value = read_mem(cpu, old_exmem["alu"])
            elif inst["op"] == "SW":
                write_mem(cpu, old_exmem["alu"], old_exmem["store_data"])
        except ValueError as exc:
            pipeline_fault(cpu, row, inst, str(exc))
            return
        new_memwb = {"inst": inst, "value": value}

    # EX: execute old ID/EX; apply forwarding.
    branch_taken = False
    branch_target = None

    if old_idex:
        inst = old_idex["inst"]
        row["EX"] = inst["text"]
        a = old_idex["a"]
        b = old_idex["b"]

        if old_exmem and writes_rd(old_exmem["inst"]) and old_exmem["inst"]["rd"] in (
            inst["rs1"], inst["rs2"]
        ):
            # A load result is not available from EX/MEM.
            if old_exmem["inst"]["op"] != "LW":
                forwarded = old_exmem["alu"]
                if inst["rs1"] == old_exmem["inst"]["rd"]:
                    a = forwarded
                    cpu["forwarding"] += 1
                    events.append("Forwarded EX/MEM result to rs1")
                if inst["rs2"] == old_exmem["inst"]["rd"] and inst["op"] in (
                    "ADD", "SUB", "AND", "OR", "BEQ", "SW"
                ):
                    b = forwarded
                    cpu["forwarding"] += 1
                    events.append("Forwarded EX/MEM result to rs2")

        if old_memwb and writes_rd(old_memwb["inst"]):
            # MEM/WB is older than EX/MEM. Never let its value overwrite a
            # newer EX/MEM forwarding result for the same source register.
            wb_rd = old_memwb["inst"]["rd"]
            exmem_is_newer_writer = bool(
                old_exmem
                and writes_rd(old_exmem["inst"])
                and old_exmem["inst"]["rd"] == wb_rd
            )
            forwarded = old_memwb["value"]
            if inst["rs1"] == wb_rd and not exmem_is_newer_writer:
                a = forwarded
                cpu["forwarding"] += 1
                events.append("Forwarded MEM/WB result to rs1")
            if (inst["rs2"] == wb_rd and inst["op"] in (
                "ADD", "SUB", "AND", "OR", "BEQ", "SW"
            ) and not exmem_is_newer_writer):
                b = forwarded
                cpu["forwarding"] += 1
                events.append("Forwarded MEM/WB result to rs2")

        op = inst["op"]
        if op in ("ADD", "SUB", "AND", "OR"):
            result = alu(op, a, b)
        elif op == "ADDI":
            result = alu(op, a, inst["imm"])
        elif op in ("LW", "SW"):
            result = alu("ADD", a, inst["imm"])
        else:
            result = None

        store_data = b
        new_exmem = {
            "inst": inst, "alu": result, "store_data": store_data
        }

        if op == "BEQ" and a == b:
            branch_taken = True
            branch_target = inst["pc"] + inst["imm"]
            events.append(f"Taken branch to byte address {branch_target}")

    # ID: decode fetched instruction, read registers.
    stall = False
    if old_ifid:
        inst = old_ifid["inst"]
        row["ID"] = inst["text"]
        if old_idex and old_idex["inst"]["op"] == "LW":
            load_rd = old_idex["inst"]["rd"]
            if load_rd != 0 and load_rd in source_regs(inst):
                stall = True
                events.append(f"Load-use stall: waiting for x{load_rd}")

        if not stall and not branch_taken:
            new_idex = {
                "inst": inst,
                "a": regs[inst["rs1"]],
                "b": regs[inst["rs2"]],
            }

    # IF: fetch unless a stall or taken branch redirects the PC.
    if stall:
        new_idex = None
        new_ifid = old_ifid
        cpu["stalls"] += 1
        row["IF"] = "HELD"
    elif branch_taken:
        new_ifid = None
        new_idex = None
        cpu["pc"] = branch_target
        cpu["flushes"] += 1
        if old_ifid:
            cpu["discarded"] += 1   # fetched wrong-path instruction thrown away
        cpu["redirects"] += 1
        events.append("Flushed younger wrong-path instructions")
        row["IF"] = "FLUSH"
    else:
        fetched = get_inst(cpu["program"], cpu["pc"])
        if fetched:
            new_ifid = {"inst": fetched, "uid": cpu["next_uid"]}
            cpu["next_uid"] += 1
            row["IF"] = fetched["text"]
            cpu["pc"] += 4
        else:
            new_ifid = None
            row["IF"] = "-"

    # Record bubbles and actual stage occupancy.
    if stall:
        events.append("Inserted one EX bubble")
    cpu["cycles"] += 1
    cpu["pipe"] = {
        "IFID": new_ifid,
        "IDEX": new_idex,
        "EXMEM": new_exmem,
        "MEMWB": new_memwb,
    }
    cpu["regs"][0] = 0
    cpu["timeline"].append(row)
    cpu["events"].append({
        "Cycle": cpu["cycles"],
        "Events": "; ".join(events) if events else "Normal pipeline operation"
    })

    # Stop only when the fetch stream and all pipeline registers are empty.
    no_pipeline_work = all(v is None for v in cpu["pipe"].values())
    no_more_fetch = get_inst(cpu["program"], cpu["pc"]) is None
    if no_pipeline_work and no_more_fetch:
        cpu["halted"] = True


def run_pipeline(cpu, max_cycles=1000):
    count = 0
    while not cpu["halted"] and count < max_cycles:
        pipe_cycle(cpu)
        count += 1
    if cpu["fault"]:
        raise RuntimeError(f"Pipeline stopped by a fault: {cpu['fault']}")
    if not cpu["halted"]:
        raise RuntimeError(
            f"Stopped after {max_cycles} cycles to prevent an infinite loop."
        )


# --------------------- EXAMPLE PROGRAM -----------------------

DEFAULT_PROGRAM = """# RISC-V example
ADDI x1, x0, 10
ADDI x2, x0, 5
ADD  x3, x1, x2
SUB  x4, x1, x2
AND  x5, x1, x2
OR   x6, x1, x2
ADDI x7, x0, 100
SW   x3, 0(x7)
LW   x8, 0(x7)
BEQ  x3, x8, 8
ADDI x9, x0, 999
ADDI x10, x0, 1
"""



# ------------------- COMPARISON HELPERS -----------------------

def run_sequential_to_completion(cpu, max_steps=1000):
    steps = 0
    while not cpu["halted"] and steps < max_steps:
        sequential_step(cpu)
        steps += 1
    if cpu["fault"]:
        raise RuntimeError(f"Sequential CPU stopped by a fault: {cpu['fault']}")
    if not cpu["halted"]:
        raise RuntimeError(
            f"Sequential execution stopped after {max_steps} instructions."
        )
    return cpu


def compare_cpu_states(seq_cpu, pipe_cpu):
    """List differences in architectural state (registers, memory, PC, count).

    Internal pipeline state and timing are expected to differ and are not
    compared.
    """
    mismatches = []
    for i, (seq_value, pipe_value) in enumerate(
        zip(seq_cpu["regs"], pipe_cpu["regs"])
    ):
        if seq_value != pipe_value:
            mismatches.append({
                "Component": f"x{i}",
                "Sequential": seq_value,
                "Pipelined": pipe_value,
            })

    all_addresses = sorted(set(seq_cpu["mem"]) | set(pipe_cpu["mem"]))
    for address in all_addresses:
        seq_value = seq_cpu["mem"].get(address, 0)
        pipe_value = pipe_cpu["mem"].get(address, 0)
        if seq_value != pipe_value:
            mismatches.append({
                "Component": f"Memory[{address}]",
                "Sequential": seq_value,
                "Pipelined": pipe_value,
            })

    for label, key in (("Instructions completed", "completed"),
                       ("Final PC", "pc")):
        if seq_cpu[key] != pipe_cpu[key]:
            mismatches.append({
                "Component": label,
                "Sequential": seq_cpu[key],
                "Pipelined": pipe_cpu[key],
            })
    return mismatches


def safe_ratio(numerator, denominator):
    """numerator / denominator, or None when the denominator is zero."""
    return numerator / denominator if denominator else None


def calculate_comparison(program):
    """Run independent fresh copies so the comparison is repeatable.

    Definitions used everywhere (all taken from the real execution engines):
      * assembled instructions  - static length of the program
      * instructions completed  - instructions that reached WB (committed);
                                  the sequential CPU completes the same set
      * pipeline cycles         - clock cycles recorded by pipe_cycle()
                                  (fill + execution + stalls + flushes + drain)
      * serial 5-stage baseline - MODEL: completed instructions x 5 cycles,
                                  i.e. a CPU that finishes one instruction's
                                  five stages before starting the next
      * CPI                     - total cycles / instructions completed
    """
    seq_cpu = make_cpu(program)
    pipe_cpu = make_cpu(program)
    run_sequential_to_completion(seq_cpu)
    run_pipeline(pipe_cpu)
    mismatches = compare_cpu_states(seq_cpu, pipe_cpu)

    instruction_count = pipe_cpu["completed"]
    serial_five_stage_cycles = instruction_count * 5
    pipeline_cycles = pipe_cpu["cycles"]
    speedup = safe_ratio(serial_five_stage_cycles, pipeline_cycles)
    cycles_saved = serial_five_stage_cycles - pipeline_cycles
    cycle_reduction_pct = (
        safe_ratio(cycles_saved, serial_five_stage_cycles) or 0
    ) * 100

    # Ideal pipeline: 4 fill cycles + 1 cycle per instruction. Anything above
    # that comes from stalls and branch-flush bubbles.
    ideal_cycles = instruction_count + 4 if instruction_count else 0
    never_executed = len(program) - len({h["PC"] for h in seq_cpu["history"]})

    if pipeline_cycles == 0:
        performance_verdict = "No pipeline cycles were recorded."
    elif cycles_saved > 0:
        performance_verdict = (
            f"The pipeline used {cycles_saved} fewer cycles than the illustrative "
            f"five-stage serial baseline ({cycle_reduction_pct:.1f}% fewer cycles)."
        )
    elif cycles_saved < 0:
        performance_verdict = (
            f"The pipeline used {abs(cycles_saved)} more cycles than the illustrative "
            f"five-stage serial baseline ({abs(cycle_reduction_pct):.1f}% more cycles). "
            "Pipeline fill/drain, stalls, and branch flushes can outweigh overlap "
            "for short programs."
        )
    else:
        performance_verdict = (
            "The pipeline and illustrative five-stage serial baseline used the same "
            "number of cycles."
        )

    return {
        "seq": seq_cpu,
        "pipe": pipe_cpu,
        "mismatches": mismatches,
        "assembled_count": len(program),
        "instruction_count": instruction_count,
        "discarded_count": pipe_cpu["discarded"],
        "never_executed": never_executed,
        "serial_five_stage_cycles": serial_five_stage_cycles,
        "pipeline_cycles": pipeline_cycles,
        "ideal_cycles": ideal_cycles,
        "extra_cycles": pipeline_cycles - ideal_cycles,
        "pipe_cpi": safe_ratio(pipeline_cycles, instruction_count),
        "pipe_ipc": safe_ratio(instruction_count, pipeline_cycles),
        "speedup": speedup,
        "cycles_saved": cycles_saved,
        "cycle_reduction_pct": cycle_reduction_pct,
        "performance_verdict": performance_verdict,
    }


# ---------------- BUILT-IN TEST BENCH (used by the Automated Tests tab) ----------------
# Each case runs the program on BOTH CPUs and checks expected registers,
# memory, completed-instruction count and (where listed) exact hazard counts.
# The same engine is also covered by the pytest suite in tests/.

BENCH_CASES = [
    dict(name="Arithmetic and logic",
         source="ADDI x1, x0, 15\nADDI x2, x0, 5\nADD x3, x1, x2\nSUB x4, x1, x2\n"
                "AND x5, x1, x2\nOR x6, x1, x2",
         regs={3: 20, 4: 10, 5: 5, 6: 15}, completed=6, stalls=0, flushes=0),
    dict(name="Loads and stores",
         source="ADDI x1, x0, 100\nADDI x2, x0, 77\nSW x2, 0(x1)\nLW x3, 0(x1)",
         regs={3: 77}, mem={100: 77}, completed=4),
    # x2 holds 5 just before the LW, so a stale register read would give 10.
    dict(name="Load-use stall",
         source="ADDI x1, x0, 100\nADDI x2, x0, 77\nSW x2, 0(x1)\nADDI x2, x0, 5\n"
                "LW x2, 0(x1)\nADD x3, x2, x2",
         regs={2: 77, 3: 154}, completed=6, stalls=1, cycles=11),
    dict(name="Taken branch and flush",
         source="ADDI x1, x0, 7\nADDI x2, x0, 7\nBEQ x1, x2, target\n"
                "ADDI x3, x0, 99\ntarget: ADDI x4, x0, 1",
         regs={3: 0, 4: 1}, completed=4, flushes=1, cycles=10),
    dict(name="Not-taken branch",
         source="ADDI x1, x0, 7\nADDI x2, x0, 8\nBEQ x1, x2, target\n"
                "ADDI x3, x0, 5\ntarget: ADDI x4, x0, 1",
         regs={3: 5, 4: 1}, completed=5, flushes=0, cycles=9),
    dict(name="Arithmetic forwarding",
         source="ADDI x1, x0, 10\nADDI x2, x0, 20\nADD x3, x1, x2\nADD x4, x3, x1",
         regs={3: 30, 4: 40}, completed=4, stalls=0, cycles=8),
    dict(name="Instruction completion count",
         source="ADDI x1, x0, 1\nADDI x2, x0, 2\nADD x3, x1, x2",
         regs={3: 3}, completed=3, cycles=7),
    dict(name="Newest forwarding result wins",
         source="ADDI x1, x0, 1\nADDI x1, x0, 2\nADD x2, x1, x1",
         regs={1: 2, 2: 4}, completed=3),
    dict(name="Load then store forwarding",
         source="ADDI x1, x0, 100\nADDI x2, x0, 55\nSW x2, 0(x1)\nLW x3, 0(x1)\n"
                "SW x3, 4(x1)\nLW x4, 4(x1)",
         regs={3: 55, 4: 55}, mem={100: 55, 104: 55}, completed=6, stalls=1),
    dict(name="Branch uses newest forwarded value",
         source="ADDI x1, x0, 1\nADDI x1, x0, 2\nBEQ x1, x0, target\n"
                "ADDI x3, x0, 9\ntarget: ADDI x4, x0, 6",
         regs={1: 2, 3: 9, 4: 6}, completed=5, flushes=0),
    dict(name="x0 remains zero",
         source="ADDI x0, x0, 99\nADDI x1, x0, 7\nADD x2, x0, x1",
         regs={0: 0, 1: 7, 2: 7}, completed=3),
    dict(name="Negative value survives store/load",
         source="ADDI x1, x0, -1\nSW x1, 0(x0)\nLW x2, 0(x0)\nBEQ x1, x2, ok\n"
                "ADDI x3, x0, 99\nok: ADDI x4, x0, 1",
         regs={1: -1, 2: -1, 3: 0, 4: 1}, completed=5, flushes=1),
    dict(name="Wrong-path store never commits",
         source="ADDI x1, x0, 5\nBEQ x0, x0, end\nSW x1, 0(x0)\nend: ADDI x2, x0, 1",
         regs={2: 1}, mem={}, completed=3, flushes=1),
]


def _bench_row(name, status, details, seq=None, pipe=None):
    return {
        "Test": name, "Status": status, "Details": details,
        "Instructions": seq["completed"] if seq else 0,
        "Pipeline cycles": pipe["cycles"] if pipe else 0,
        "Stalls": pipe["stalls"] if pipe else 0,
        "Flushes": pipe["flushes"] if pipe else 0,
        "Forwarding": pipe["forwarding"] if pipe else 0,
    }


def run_bench():
    """Run every built-in test; returns a list of result rows (dicts)."""
    rows = []
    for case in BENCH_CASES:
        name = case["name"]
        seq = pipe = None
        try:
            program = clean_source(case["source"])
            seq, pipe = make_cpu(program), make_cpu(program)
            run_sequential_to_completion(seq)
            run_pipeline(pipe)
            issues = []
            for label, cpu in (("Sequential", seq), ("Pipeline", pipe)):
                for reg, expected in case["regs"].items():
                    if cpu["regs"][reg] != expected:
                        issues.append(
                            f"{label} x{reg}: expected {expected}, got {cpu['regs'][reg]}")
                if "mem" in case and cpu["mem"] != case["mem"]:
                    issues.append(
                        f"{label} memory: expected {case['mem']}, got {cpu['mem']}")
                if cpu["completed"] != case["completed"]:
                    issues.append(
                        f"{label} completed: expected {case['completed']}, "
                        f"got {cpu['completed']}")
            for key in ("stalls", "flushes", "cycles"):
                if key in case and pipe[key] != case[key]:
                    issues.append(
                        f"Pipeline {key}: expected {case[key]}, got {pipe[key]}")
            for m in compare_cpu_states(seq, pipe):
                issues.append(
                    f"Sequential/pipeline mismatch in {m['Component']}: "
                    f"{m['Sequential']} vs {m['Pipelined']}")
            rows.append(_bench_row(
                name, "PASS" if not issues else "FAIL",
                "All checks passed" if not issues else "; ".join(issues), seq, pipe))
        except Exception as exc:  # report the failing case, never crash the tab
            rows.append(_bench_row(name, "FAIL", f"{type(exc).__name__}: {exc}", seq, pipe))

    # --- machine code ---
    try:
        word = clean_source("ADD x3, x1, x2")[0]["word"]
        decoded = decode_word(word)
        ok = (word == 0x002081B3 and decoded["op"] == "ADD"
              and (decoded["rd"], decoded["rs1"], decoded["rs2"]) == (3, 1, 2))
        rows.append(_bench_row(
            "Machine-code encode/decode", "PASS" if ok else "FAIL",
            f"Encoded 0x{word:08X} (expected 0x002081B3); decoded {decoded}"))
    except Exception as exc:
        rows.append(_bench_row("Machine-code encode/decode", "FAIL", str(exc)))

    try:
        mismatched = []
        for text in ("ADDI x1, x2, -2048", "LW x3, 2044(x4)", "SW x5, -4(x6)",
                     "BEQ x7, x8, 4092", "SUB x9, x10, x11"):
            inst = clean_source(text)[0]
            back = decode_word(inst["word"])
            if any(back[k] != inst[k] for k in ("op", "rd", "rs1", "rs2", "imm")):
                mismatched.append(text)
        rows.append(_bench_row(
            "Machine-code round trip", "PASS" if not mismatched else "FAIL",
            "All instruction formats round-trip" if not mismatched
            else f"Round trip failed for: {mismatched}"))
    except Exception as exc:
        rows.append(_bench_row("Machine-code round trip", "FAIL", str(exc)))

    # --- error handling ---
    try:
        parse_reg("x32")
        ok, detail = False, "Invalid register x32 was accepted"
    except ValueError:
        ok, detail = True, "Invalid register rejected correctly"
    rows.append(_bench_row("Invalid register validation", "PASS" if ok else "FAIL", detail))

    bad = ["ADD x1, x2", "LW x1, 4, x2", "MUL x1, x2, x3", "BEQ x1, x2, 2"]
    accepted = []
    for text in bad:
        try:
            clean_source(text)
            accepted.append(text)
        except ValueError:
            pass
    rows.append(_bench_row(
        "Malformed assembly rejected", "PASS" if not accepted else "FAIL",
        "All malformed lines rejected with an error message" if not accepted
        else f"Accepted invalid lines: {accepted}"))

    # --- memory fault safety ---
    try:
        program = clean_source("ADDI x1, x0, 2\nLW x2, 0(x1)")
        seq, pipe = make_cpu(program), make_cpu(program)
        for _ in range(20):
            sequential_step(seq)
            pipe_cycle(pipe)
        ok = (seq["fault"] and pipe["fault"] and seq["halted"] and pipe["halted"]
              and seq["completed"] == pipe["completed"] == 1)
        rows.append(_bench_row(
            "Unaligned load halts safely", "PASS" if ok else "FAIL",
            "Both CPUs halted with a fault after 1 completed instruction" if ok
            else f"seq fault={seq['fault']!r}, pipe fault={pipe['fault']!r}, "
                 f"completed {seq['completed']}/{pipe['completed']}", seq, pipe))
    except Exception as exc:
        rows.append(_bench_row("Unaligned load halts safely", "FAIL", str(exc)))
    return rows
