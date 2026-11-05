import pytest

from riscv_engine import (
    clean_source, encode_instruction, decode_word, parse_reg, parse_imm,
)


def one(text):
    return clean_source(text)[0]


# ---------------------------- assembler ----------------------------

def test_assembles_all_supported_instructions():
    program = clean_source(
        "ADD x3, x1, x2\nSUB x4, x1, x2\nAND x5, x1, x2\nOR x6, x1, x2\n"
        "ADDI x1, x0, -5\nLW x2, 8(x3)\nSW x2, -4(x3)\nBEQ x1, x2, 8"
    )
    assert [i["op"] for i in program] == [
        "ADD", "SUB", "AND", "OR", "ADDI", "LW", "SW", "BEQ"]
    assert [i["pc"] for i in program] == [0, 4, 8, 12, 16, 20, 24, 28]
    lw, sw = program[5], program[6]
    assert (lw["rd"], lw["rs1"], lw["imm"]) == (2, 3, 8)
    assert (sw["rs2"], sw["rs1"], sw["imm"]) == (2, 3, -4)


def test_comments_blank_lines_case_and_labels():
    program = clean_source(
        "# header\n\n  addi X1, x0, 0x10   # trailing\nloop:\n"
        "BEQ x1, x1, end // jump\nADDI x2, x0, 1\nend: ADDI x3, x0, 0b11\n"
    )
    assert [i["pc"] for i in program] == [0, 4, 8, 12]
    assert program[0]["imm"] == 16
    assert program[1]["imm"] == 8           # label 'end' is at 12, BEQ is at 4
    assert program[3]["imm"] == 3


def test_empty_and_comment_only_source_gives_empty_program():
    assert clean_source("") == []
    assert clean_source("# nothing\n\n// still nothing") == []


@pytest.mark.parametrize("text, fragment", [
    (",", "Unsupported instruction"),
    ("ADD x1, x2", "Expected rd, rs1, rs2"),
    ("ADD x1, x2 x3", "Expected rd, rs1, rs2"),     # missing comma
    ("ADD x1, x2, x3,", "Empty operand"),
    ("ADD x1,, x3", "Empty operand"),
    ("ADD x1, x2, x32", "Register out of range"),
    ("ADD x1, x2, y3", "Invalid register"),
    ("ADD x1, x2, 5", "Invalid register"),
    ("ADDI x1, x0", "Expected rd, rs1, immediate"),
    ("ADDI x1, x0, abc", "Invalid immediate"),
    ("ADDI x1, x0, 1_0", "Invalid immediate"),
    ("ADDI x1, x0, 2048", "does not fit in 12 signed bits"),
    ("ADDI x1, x0, -2049", "does not fit in 12 signed bits"),
    ("LW x1, 4, x2", "Expected register, offset(base)"),
    ("LW x1, 0(x2", "offset(base)"),
    ("LW x1, 2(x2)", "multiple of 4"),
    ("SW x1, 4096(x2)", "does not fit in 12 signed bits"),
    ("LW x1, 0(x40)", "Register out of range"),
    ("BEQ x1, x2", "Expected rs1, rs2"),
    ("BEQ x1, x2, 2", "multiple of 4"),
    ("BEQ x1, x2, -8", "before the start of the program"),
    ("BEQ x1, x2, nowhere", "unknown label"),
    ("BEQ x1, x2, 4096", "does not fit in 13 signed bits"),
    ("MUL x1, x2, x3", "Unsupported instruction"),
    ("JAL x1, 4", "Unsupported instruction"),
])
def test_malformed_input_gives_clear_assembly_error(text, fragment):
    with pytest.raises(ValueError) as info:
        clean_source(text)
    message = str(info.value)
    assert fragment in message
    assert "Assembly error on line 1" in message


def test_label_is_not_allowed_as_an_addi_immediate():
    with pytest.raises(ValueError, match="Invalid immediate"):
        clean_source("loop: ADDI x1, x0, loop")


def test_duplicate_and_invalid_labels():
    with pytest.raises(ValueError, match="duplicate label"):
        clean_source("a: ADDI x1, x0, 1\na: ADDI x2, x0, 2")
    with pytest.raises(ValueError, match="invalid label"):
        clean_source("1bad: ADDI x1, x0, 1")


def test_error_reports_the_source_line_number():
    with pytest.raises(ValueError, match="line 3"):
        clean_source("ADDI x1, x0, 1\n\nFOO x1\n")


def test_branch_to_end_of_program_is_allowed():
    program = clean_source("BEQ x0, x0, end\nADDI x1, x0, 1\nend:")
    assert program[0]["imm"] == 8


def test_parse_helpers():
    assert parse_reg(" X31 ") == 31
    assert parse_imm("-0x10") == -16
    assert parse_imm("+7") == 7
    assert parse_imm("010") == 10     # decimal, not octal


# -------------------------- machine code ---------------------------

# Encodings below are the standard RV32I encodings (checked against the
# RISC-V specification / a reference assembler listing).
KNOWN = [
    ("ADD x3, x1, x2", 0x002081B3),
    ("SUB x4, x1, x2", 0x40208233),
    ("AND x5, x1, x2", 0x0020F2B3),
    ("OR x6, x1, x2", 0x0020E333),
    ("ADDI x1, x0, 10", 0x00A00093),
    ("ADDI x1, x0, -1", 0xFFF00093),
    ("LW x8, 0(x7)", 0x0003A403),
    ("SW x3, 0(x7)", 0x0033A023),
    ("BEQ x3, x8, 8", 0x00818463),
    ("BEQ x0, x0, 0", 0x00000063),
]


@pytest.mark.parametrize("text, word", KNOWN)
def test_known_encodings(text, word):
    inst = one(text)
    assert inst["word"] == word == encode_instruction(inst)


def test_the_documented_example_word():
    assert one("ADD x3, x1, x2")["word"] == 0x002081B3
    d = decode_word(0x002081B3)
    assert (d["op"], d["rd"], d["rs1"], d["rs2"]) == ("ADD", 3, 1, 2)


def fields(inst):
    return {k: inst[k] for k in ("op", "rd", "rs1", "rs2", "imm")}


@pytest.mark.parametrize("text", [
    "ADD x31, x30, x29", "SUB x1, x2, x3", "AND x0, x1, x2", "OR x31, x0, x31",
    "ADDI x1, x2, 2047", "ADDI x1, x2, -2048", "ADDI x0, x0, 0",
    "LW x31, 2044(x30)", "LW x1, -2048(x2)", "LW x1, 0(x0)",
    "SW x31, 2044(x30)", "SW x1, -2048(x2)", "SW x0, 4(x0)", "SW x5, -4(x6)",
    "BEQ x1, x2, 4092", "BEQ x31, x30, 4", "BEQ x0, x0, 0",
])
def test_encode_then_decode_round_trip(text):
    inst = one(text)
    assert fields(decode_word(inst["word"])) == fields(inst)


def test_most_negative_branch_offset_round_trips():
    padding = "ADDI x0, x0, 0\n" * 1024          # BEQ ends up at address 4096
    inst = clean_source(padding + "BEQ x1, x2, -4096")[-1]
    assert inst["imm"] == -4096
    assert fields(decode_word(inst["word"])) == fields(inst)


def test_sw_and_branch_immediates_are_split_correctly():
    # negative store offset spans both immediate fields
    d = decode_word(one("SW x1, -4(x2)")["word"])
    assert d["imm"] == -4
    # branch offsets that exercise bits 11, 12 and 4:1
    for offset in (4, 8, 2048, 2052, 4092):
        d = decode_word(one(f"BEQ x1, x2, {offset}")["word"])
        assert d["imm"] == offset


def test_decode_rejects_unsupported_words():
    for word in (0x00000000, 0xFFFFFFFF, 0x00001063,  # BNE
                 0x00209033,                          # SLL
                 0x40207033,                          # funct7=0x20 with AND
                 0x00001003,                          # LH
                 0x00001023):                         # SH
        with pytest.raises(ValueError):
            decode_word(word)


def test_decode_ignores_bits_above_32():
    assert decode_word(0x1_002081B3)["op"] == "ADD"


@pytest.mark.parametrize("text", [
    "ADD x3, x1, x2", "SUB x4, x1, x2", "AND x5, x6, x7", "OR x8, x9, x10",
    "ADDI x1, x2, -5", "LW x3, 8(x4)", "SW x5, -4(x6)", "BEQ x1, x2, 12",
])
def test_disassemble_reassembles_to_the_same_machine_word(text):
    from riscv_engine import disassemble
    word = one(text)["word"]
    assert disassemble(decode_word(word)) == text
    assert clean_source(disassemble(decode_word(word)))[0]["word"] == word
