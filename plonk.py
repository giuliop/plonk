"""This module implements the PLONK zero-knowledge proof system,
   using the KZG polynomial commitment scheme."""

import secrets
import hashlib
import sympy

import kzg
import circuit_compiler as cc
import utils


def run_setup(circuit_filepath):
    """Preprocess the circuit and generate all the common input for the
       prover and the verifier.

    Args:
        circuit_file: path of the file containing the circuit description
                      in a format parseable by the circuit_compiler module
    Returns:
        A dict with the the common preprocessed input for prover and verifier:
        - gp: the trusted setup paramenters of the KZG commitment scheme
        - circuit: a dict with the polynomials QL, QR, QO, QM, QC describing
                   the gates and the permutation vectors a, b, c, as lists
        - num_gates: the number of gates
        - order: the curve order
    """
    circuit = cc.parse_file(circuit_filepath)
    n = circuit['num_gates']
    q = kzg.curve.curve_order
    gp = kzg.trusted_setup(n + 5)

    w, w_order = utils.find_nth_root(n, q, kzg.factorization)
    H = [pow(w, i, q) for i in range(w_order)]
    k1, k2 = utils.get_cosets_constants(w, H, q)
    H1 = [k1 * h % q for h in H]
    H2 = [k2 * h % q for h in H]

    # TODO: if w_order > num_gates since w_order is the smallest power of 2
    #       greater than num_gates, we need to pad the circuit
    a, b, c = build_permutations(H, H1, H2, circuit)

    QL = utils.interpolate(H, circuit['qL'], q)
    QR = utils.interpolate(H, circuit['qR'], q)
    QO = utils.interpolate(H, circuit['qO'], q)
    QM = utils.interpolate(H, circuit['qM'], q)
    QC = utils.interpolate(H, circuit['qC'], q)

    circuit =  {'QL': QL, 'QR': QR, 'QO': QO, 'QM': QM, 'QC': QC,
               'a': a, 'b': b, 'c': c}

    return {'gp': gp, 'circuit': circuit, 'num_gates': n, 'order': q}


def build_permutations(H, H1, H2, circuit):
    """Build the copy constraints vectors a, b, c representing the
       sigma permutation constraints of the plonk paper.

    Args:
        H: a list of the evaluation points
        H1: a list of the first coset of the evaluation points
        H2: a list of the second coset of the evaluation points
        circuit: the circuit description as a dict with the vectors
                 a, b, c as lists of integers representing the variables
                 of the the gates (same int, same variable)

    Returns:
        The three vectors a, b, c encoding the copy constraints
        as lists of integers

    """
    a, b, c = circuit['a'], circuit['b'], circuit['c']
    n = circuit['num_gates']
    abc = a + b + c
    new_abc = H + H1 + H2

    for i, x in enumerate(abc):
        for j in range(i+1, len(abc)):
            if x == abc[j]:
                new_abc[i], new_abc[j] = new_abc[j], new_abc[i]
                break

    a = new_abc[:n]
    b = new_abc[n:2*n]
    c = new_abc[2*n:]

    return a, b, c


def generate_proof(setup, public_inputs, trace, H, H1, H2):
    """Generates a proof for the given circuit and witness.

    Args:
        setup: A dict with the common input for the prover and verifier
               coming from the run_setup function.
        public_inputs: A list with the public inputs for the proof
        trace: a dict with fields 'left', 'right', 'output' as lists
               of the inputs and outputs values of the gates
        H: a list of the evaluation points
        H1: a list of the first coset of the evaluation points
        H2: a list of the second coset of the evaluation points

    Returns:

    """
    transcript = str(setup) + str(public_inputs)
    q = setup['order']  # the curve order
    gp = setup['gp']  # the trusted setup parameters

    # ROUND 1
    # Commit to the polynomials A, B, C representing the left, right
    # and output values of the gates, randominzed for zero-knowledge
    randoms = [secrets.randbelow(q) for _ in range(9)]

    x = sympy.symbols('x')
    domain = sympy.FF(q)
    coeff_Zh = [1] + [0] * (len(H) - 1) + [-1]
    Zh = sympy.Poly(coeff_Zh, x, domain=domain)

    A = utils.interpolate(H, trace.left, q)
    B = utils.interpolate(H, trace.rigth, q)
    C = utils.interpolate(H, trace.output, q)

    A = (sympy.Poly(randoms[0] * x + randoms[1], x, domain=domain)
         * Zh + sympy.Poly(A, x, domain=domain))

    B = (sympy.Poly(randoms[2] * x + randoms[3], x, domain=domain)
         * Zh + sympy.Poly(B, x, domain=domain))

    C = (sympy.Poly(randoms[4] * x + randoms[5], x, domain=domain)
         * Zh + sympy.Poly(C, x, domain=domain))

    com_A = kzg.commit(A, gp)
    com_B = kzg.commit(B, gp)
    com_C = kzg.commit(C, gp)

    transcript += str(com_A) + str(com_B) + str(com_C)

    # ROUND 2
    # Commit to polynomial Z, which eccodes the permutation constraints
    beta = fiat_shamir([transcript, 0], q)
    gamma = fiat_shamir([transcript, 1], q)

    left, right, output = trace.left, trace.right, trace.output
    a, b, c = [setup['circuit'][x] for x in ['a', 'b', 'c']]
    acc = [1]

    for i in range(1, setup['num_gates']):
        num = sympy.Mod((left[i-1] + beta * H[i-1] + gamma)
               * (right[i-1] + beta * H1[i-1] + gamma)
               * (output[i-1] + beta * H2[i-1] + gamma), q)

        den = sympy.Mod((left[i-1] + beta * a[i-1] + gamma)
               * (right[i-1] + beta * b[i-1] + gamma)
               * (output[i-1] + beta * c[i-1] + gamma), q)

        acc.append(sympy.Mod(acc[i-1] * num * sympy.mod_inverse(den, q),  q))

    acc_poly = utils.interpolate(H, acc, q)

    Z = (sympy.Poly([randoms[6], randoms[7], randoms[8]], x, domain=domain)
         * Zh + sympy.Poly(acc_poly, x, domain=domain))

    com_Z = kzg.commit(Z, gp)

    # ROUND 3




def fiat_shamir(data, p):
    """Hashes data and reduces it modulo p.

    Args:
        data: The data to hash, should be ok to
              serialize it to a string with str().
        p: The prime number to reduce the hash to.

    Returns:
        An integer modulo p.
    """

    serialized_data = str(data).encode('utf-8')
    hasher = hashlib.blake2b(serialized_data)
    digest_bytes = hasher.digest()

    digest_int = int.from_bytes(digest_bytes, 'big')
    return digest_int % p


def test_plonk_by_hand():
    """Let's replicate:
       https://research.metastate.dev/plonk-by-hand-part-2-the-proof/
    """
    # SETUP
    circuit = cc.parse_file('circuit_example')
    q = 17  # prime field
    w = 4   # generator of order 4 in p
    k1, k2 = 2, 3  # to generate the cosets from the generator w
    H = [pow(w, i, q) for i in range(4)]
    assert H == [1, 4, 16, 13]
    H1 = [k1 * x % q for x in H]
    assert H1 == [2, 8, 15, 9]
    H2 = [k2 * x % q for x in H]
    assert H2 == [3, 12, 14, 5]

    a, b, c = build_permutations(H, H1, H2, circuit)

    QL = utils.interpolate(H, circuit['qL'], q)
    QR = utils.interpolate(H, circuit['qR'], q)
    QO = utils.interpolate(H, circuit['qO'], q)
    QM = utils.interpolate(H, circuit['qM'], q)
    QC = utils.interpolate(H, circuit['qC'], q)

    n = circuit['num_gates']
    circuit =  {'QL': QL, 'QR': QR, 'QO': QO, 'QM': QM, 'QC': QC,
               'a': a, 'b': b, 'c': c}

    setup = {'gp': {}, 'circuit': circuit, 'num_gates': n, 'order': q}

    assert circuit['QL'] == [16, 4, 1, 13]
    assert circuit['QR'] == [16, 4, 1, 13]
    assert circuit['QO'] == [16]
    assert circuit['QM'] == [1, 13, 16, 5]
    assert circuit['QC'] == [0]
    assert circuit['a'] == [2, 8, 15, 3]
    assert circuit['b'] == [1, 4, 16, 12]
    assert circuit['c'] == [13, 9, 5, 14]

    # PROVER

    # ROUND 1
    left = [3, 4, 5, 9]
    right = [3, 4, 5, 16]
    output = [9, 16, 25, 25]

    A = utils.interpolate(H, left, q)
    B = utils.interpolate(H, right, q)
    C = utils.interpolate(H, output, q)

    assert A == [3, 3, 13, 1]
    assert B == [13, 14, 3, 7]
    assert C == [4, 11, 5, 6]

    x = sympy.symbols('x')
    domain = sympy.FF(q)
    coeff_Zh = [1] + [0] * (len(H) - 1) + [-1]
    Zh = sympy.Poly(coeff_Zh, x, domain=domain)

    randoms = [7, 4, 11, 12, 16, 2]

    A = (sympy.Poly(randoms[0] * x + randoms[1], x, domain=domain)
         * Zh + sympy.Poly(A, x, domain=domain))

    B = (sympy.Poly(randoms[2] * x + randoms[3], x, domain=domain)
         * Zh + sympy.Poly(B, x, domain=domain))

    C = (sympy.Poly(randoms[4] * x + randoms[5], x, domain=domain)
         * Zh + sympy.Poly(C, x, domain=domain))

    assert [x % q for x in A.all_coeffs()] == [7, 4, 3, 3, 6, 14]
    assert [x % q for x in B.all_coeffs()] == [11, 12, 13, 14, 9, 12]
    assert [x % q for x in C.all_coeffs()] == [16, 2, 4, 11, 6, 4]

    com_A = (91, 66)
    com_B = (26, 45)
    com_C = (91, 35)

    # ROUND 2
    beta = 12
    gamma = 13
    acc = [1]

    for i in range(1, n):
        num = sympy.Mod((left[i-1] + beta * H[i-1] + gamma)
               * (right[i-1] + beta * H1[i-1] + gamma)
               * (output[i-1] + beta * H2[i-1] + gamma), q)

        den = sympy.Mod((left[i-1] + beta * a[i-1] + gamma)
               * (right[i-1] + beta * b[i-1] + gamma)
               * (output[i-1] + beta * c[i-1] + gamma), q)

        acc.append(sympy.Mod(acc[i-1] * num * sympy.mod_inverse(den, q),  q))
    assert acc == [1, 3, 9, 4]

    acc_poly = utils.interpolate(H, acc, q)
    assert acc_poly == [14, 5, 16, 0]

    randoms += [14, 11, 7]

    Z = (sympy.Poly([randoms[6], randoms[7], randoms[8]], x, domain=domain)
         * Zh + sympy.Poly(acc_poly, x, domain=domain))
    assert [x % q for x in Z.all_coeffs()] == [14, 11, 7, 14, 8, 5, 10]

    com_Z = (32, 59)

if __name__ == '__main__':
    test_plonk_by_hand()