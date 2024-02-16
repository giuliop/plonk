"""Compile Plonk circuits into the parameters for the zk-SNARK.

Circuit source files should be text files.
Lines starting with the character `#` are considered comments and are ignored.

A circuit with n gates is represented as a list of n lines, each either

    - a declaration of a public input in the form:
         public x_i
      where i, j, k are numbers between 1 and 3n

    - an addition gate in the form:
         x_i + x_j = x_k
      where i, j, k are numbers between 1 and 3n

    - a multiplication gate in the form:
         x_i * x_j = x_k
      where i, j, k are numbers between 1 and 3n
"""

import re

def parse_file(file_path, padding=True):
    """Parse the circuit source file into the vectors:
       - a, b, c to encode the left, right input,output of each gate.
       - ql, qr, qo, qm, qc to encode the gates operations so that we have:
           ql[i] * a[i] + qr[i] * b[i] + qo[i] * c[i] + qm[i] * a[i] * b[i] + qc[i] = 0

       The vectors a, b, c outputted the copy constraints for the circuit.

       If padding is True, the circuit is padded with empty gates so that the
       number of gates is a power of 2.

    Args:
        file_path: string path to the circuit source file
        padding: boolean to indicate if the circuit should be padded

    Returns:
        A dictionary with the following keys:
        - num_gates: the number of gates
        - ql, qr, qo, qm, qc, a, b, c: the output vectors as lists

    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [stripped_line for line in file
                 if (stripped_line := line.strip())
                    and not stripped_line.startswith("#")]

    ql, qr, qo, qm, qc, a, b, c = ([] for _ in range(8))

    op_mappings = {
            "*": (0, 0, -1, 1, 0),
            "+": (1, 1, -1, 0, 0),
            "public": (1, 0, 0, 0, 0)
    }

    n = len(lines)
    for line in lines:
        try:
            a_i, b_i, c_i, op = parse_line(line, n)
        except ValueError as e:
            raise ValueError(f"Error parsing line '{line}'") from e

        a.append(a_i)
        b.append(b_i)
        c.append(c_i)

        values = op_mappings[op]
        ql.append(values[0])
        qr.append(values[1])
        qo.append(values[2])
        qm.append(values[3])
        qc.append(values[4])

    if padding:
        old_n = n
        n = next_power_of_2(old_n)
        for _ in range(n - old_n):
            for x in [a, b, c, ql, qr, qo, qm, qc]:
                x.append(0)

    return {"num_gates": n, "ql": ql, "qr": qr, "qo": qo,
            "qm": qm, "qc": qc, "a": a, "b": b, "c": c}

def parse_line(line, n):
    """Parse a line of the circuit source file.

    A valid line has one of these two forms:
        public x_i
        x_i + x_j = x_k
        x_i * x_j = x_k
    where i, j, k are numbers between 1 and 3*n

    Args:
        line (str): a line of the circuit source file
        n (int): the number of gates in the circuit

    Returns:
        a tuple (a, b, c, op) where:
        - a, b, c are the indices of the variables in the circuit
        - op is the operation of the gate: '+' , '*' or 'public'
    """
    if line.startswith("public"):
        pattern = r"public x_(\d+)$"
        match = re.match(pattern, line)
        if not match:
            raise ValueError("Invalid line format")

        a = int(match.group(1))

        if not (1 <= a <= 3*n):
            raise ValueError("Variable index out of allowed range")

        return a, 0, a, "public"

    # Regex for addition and multiplication gates
    pattern = r"x_(\d+) (\+|\*) x_(\d+) = x_(\d+)$"
    match = re.match(pattern, line)
    if not match:
        raise ValueError("Invalid line format")

    a, op, b, c = match.groups()
    a, b, c = int(a), int(b), int(c)

    if not (1 <= a <= 3*n and 1 <= b <= 3*n and 1 <= c <= 3*n):
        raise ValueError("Variable indices out of allowed range")

    return a, b, c, op

def next_power_of_2(n):
    # If n is already a power of 2, return n
    if n and not (n & (n - 1)):
        return n

    # Find the position of the most significant bit
    msb_pos = n.bit_length()

    # Construct the next power of 2 using left shift
    return 1 << msb_pos

def test_parse_file():
    """Test the parse_file function."""
    path = "circuits/pythagoras_c_private"
    # path = "circuits/pythagoras_abc_private"
    print("\nInput file:\n")
    with open(path, 'r', encoding='utf-8') as file:
        print(file.read())

    print("\nOutput:\n")
    output = parse_file(path, padding=True)
    for k, v in output.items():
        print(f"{k}: {v}")
    print()

    # assert output == {'num_gates': 4, 'ql': [0, 0, 0, 1], 'qr': [0, 0, 0, 1],
                      # 'qo': [-1, -1, -1, -1], 'qm': [1, 1, 1, 0],
                      # 'qc': [0, 0, 0, 0], 'a': [1, 3, 5, 2],
                      # 'b': [1, 3, 5, 4], 'c': [2, 4, 6, 6]}

if __name__ == "__main__":
    test_parse_file()
