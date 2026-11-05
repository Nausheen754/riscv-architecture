
class RISCVProcessor:
    def __init__(self):
        self.registers = [0] * 32
        self.memory = {}
        self.pc = 0
        self.program = []

    def load_program(self, program):
        self.program = program
        self.registers = [0] * 32
        self.memory = {}
        self.pc = 0

    def get_value(self, operand):
        operand = operand.strip()

        if operand.startswith("x"):
            return self.registers[int(operand[1:])]

        return int(operand)

    def execute_instruction(self, instruction):
        parts = instruction.replace(",", "").split()

        if not parts:
            self.pc += 4
            return

        opcode = parts[0].upper()
        next_pc = self.pc + 4
        result = None

        if opcode == "ADDI":
            rd = int(parts[1][1:])
            rs1 = int(parts[2][1:])
            immediate = int(parts[3])

            result = self.registers[rs1] + immediate

            if rd != 0:
                self.registers[rd] = result

        elif opcode in ("ADD", "SUB", "AND", "OR"):
            rd = int(parts[1][1:])
            rs1 = int(parts[2][1:])
            rs2 = int(parts[3][1:])

            a = self.registers[rs1]
            b = self.registers[rs2]

            if opcode == "ADD":
                result = a + b
            elif opcode == "SUB":
                result = a - b
            elif opcode == "AND":
                result = a & b
            elif opcode == "OR":
                result = a | b

            if rd != 0:
                self.registers[rd] = result

        elif opcode in ("LW", "SW"):
            reg = int(parts[1][1:])
            offset, base = parts[2].split("(")
            base = base.rstrip(")")
            address = self.registers[int(base[1:])] + int(offset)

            if address % 4 != 0:
                raise ValueError("Address must be 4-byte aligned")

            if opcode == "LW":
                result = self.memory.get(address, 0)

                if reg != 0:
                    self.registers[reg] = result
            else:
                result = self.registers[reg]
                self.memory[address] = result

        elif opcode == "BEQ":
            rs1 = int(parts[1][1:])
            rs2 = int(parts[2][1:])
            offset = int(parts[3])

            if self.registers[rs1] == self.registers[rs2]:
                next_pc = self.pc + offset

        else:
            raise ValueError(f"Unsupported instruction: {opcode}")

        # x0 is always zero.
        self.registers[0] = 0
        self.pc = next_pc

        return result

    def run(self, max_steps=100):
        print("=== RISC-V Processor Simulator ===")
        steps = 0

        while 0 <= self.pc < len(self.program) * 4:
            if steps >= max_steps:
                print("Stopped: instruction limit reached.")
                break

            instruction = self.program[self.pc // 4]

            print(f"\nPC = {self.pc}")
            print(f"Instruction = {instruction}")

            try:
                result = self.execute_instruction(instruction)
                print(f"Result = {result}")
            except (ValueError, IndexError) as error:
                print("Execution error:", error)
                break

            steps += 1

        print("\n=== Final Registers ===")
        for i, value in enumerate(self.registers):
            if value != 0:
                print(f"x{i} = {value}")

        print("\n=== Data Memory ===")
        print(self.memory)
        print("Final PC =", self.pc)


if __name__ == "__main__":
    cpu = RISCVProcessor()

    program = [
        "ADDI x1, x0, 10",
        "ADDI x2, x0, 5",
        "ADD x3, x1, x2",
        "SUB x4, x1, x2",
        "AND x5, x1, x2",
        "OR x6, x1, x2",
        "ADDI x7, x0, 100",
        "SW x3, 0(x7)",
        "LW x8, 0(x7)",
        "BEQ x3, x8, 8",
        "ADDI x9, x0, 999",
        "ADDI x10, x0, 1"
    ]

    cpu.load_program(program)
    cpu.run()
