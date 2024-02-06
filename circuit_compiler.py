"""Compile Plonk circuits into the parameters for the zk-SNARK.

Circuit source files should be text files.
A circuit with n gates is represented as a list of lines, each either
    - a comment line starting with  the character `#`

    - an addition gate in the form: x_i + x_j = x_k
      where i, j, k are numbers between 1 and 3n

    - a multiplication gate in the form: x_i * x_j = x_k
      where i, j, k are numbers between 1 and 3n
"""

def parse_file(file_path):
    """Parse the circuit source file into the vectors:
       - qL, qR, qO, qM, qC to encode the gates operations
       - a, b, c to encode the left, right input,output of each gate.
       so that for each gate i, we have:
           qL[i] * a[i] + qR[i] * b[i] + qO[i] * c[i] + qM[i] * a[i] * b[i] + qC[i] = 0
        but the vectors a[i], b[i], c[i] are the indices of the variables in the circuit,
        not the actual values.

    Args:
        file_path (str): the path to the circuit source file

    Returns:
        A dictionary with the following keys:
        - n: the number of gates
        - qL, qR, qO, qM, qC, a, b, c: the outpur vectors as lists

    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = [stripped_line for line in file
                 if (stripped_line := line.strip())
                    and not stripped_line.startswith("#")]

    qL, qR, qO, qM, qC, a, b, c = ([] for _ in range(8))

    op_mappings = {
            "*": (0, 0, -1, 1, 0),
            "+": (1, 1, -1, 0, 0),
    }

    for line in lines:
        try:
            a_i, b_i, c_i, op = parse_line(line, len(lines))
        except ValueError as e:
            raise ValueError(f"Error parsing line '{line}'") from e

        a.append(a_i)
        b.append(b_i)
        c.append(c_i)

        values = op_mappings[op]
        qL.append(values[0])
        qR.append(values[1])
        qO.append(values[2])
        qM.append(values[3])
        qC.append(values[4])

    return {"n": len(lines), "qL": qL, "qR": qR, "qO": qO, "qM": qM,
            "qC": qC, "a": a, "b": b, "c": c}

def parse_line(line, n):
    """Parse a line of the circuit source file.

       A valid line has one of these two forms:
           x_i + x_j = x_k
           x_i * x_j = x_k
       where i, j, k are numbers between 1 and n

    Args:
        line (str): a line of the circuit source file
        n : the number of gates in the circuit

    Returns:
        a tuple (a, b, c, op) where:
        - a, b, c are the indices of the variables in the circuit
        - op is the operation of the gate
    """
    try:
        assert isinstance(line, str)
        assert isinstance(n, int)
        assert n > 0
        elements = line.split()
        assert len(elements) == 5
        a, op, b, equal, c = elements
        assert equal == "="
        assert op in ["+", "*"]
        assert a.startswith("x_") and b.startswith("x_") and c.startswith("x_")
        a = int(a[2:])
        b = int(b[2:])
        c = int(c[2:])
        assert 1 <= a <= 3*n
        assert 1 <= b <= 3*n
        assert 1 <= c <= 3*n
    except (AssertionError, ValueError) as e:
        raise ValueError("Invalid line format") from e

    return a, b, c, op

def test_parse_file():
    """Test the parse_file function."""
    path = "circuit_example"
    print("\nInput file:\n")
    with open(path, 'r', encoding='utf-8') as file:
        print(file.read())

    print("\nOutput:\n")
    output = parse_file(path)
    for k, v in output.items():
        print(f"{k}: {v}")
    print()

    assert output == {'n': 4, 'qL': [0, 0, 0, 1], 'qR': [0, 0, 0, 1],
                      'qO': [-1, -1, -1, -1], 'qM': [1, 1, 1, 0],
                      'qC': [0, 0, 0, 0], 'a': [1, 3, 5, 2],
                      'b': [1, 3, 5, 4], 'c': [2, 4, 6, 6]}

if __name__ == "__main__":
    test_parse_file()
