"""This module implements the PLONK zero-knowledge proof system,
   using the KZG polynomial commitment scheme."""

import secrets
import sympy

import kzg
import circuit_compiler as cc
import utils


def Poly(expr):
    """Convenience function to create sympy polynomials in x
       in the domain of field_order

    Args:
        expr: a list of coefficients starting from highest degree or
              a sympy expression with the symbol x (e.g., x**2 - 1)
    Returns:
        A sympy Poly built from expr
    """
    domain = sympy.FF(kzg.curve.curve_order)
    if isinstance(expr, list):
        return sympy.Poly(expr, sympy.symbols('x'), domain=domain)
    return sympy.Poly(expr, domain=domain)

def run_setup(circuit_filepath):
    """Preprocess the circuit and generate all the common input for the
       prover and the verifier.

    Args:
        circuit_file: path of the file containing the circuit description
                      in a format parseable by the circuit_compiler module
    Returns:
        A dict with the the common preprocessed input for prover and verifier:
        - gp: the trusted setup paramenters of the KZG commitment scheme
        - circuit: a dict with the coefficients for the polynomials
                   QL, QR, QO, QM, QC describing the gates and the
                   permutation vectors a, b, c, all as lists
        - num_gates: the number of gates
        - order: the curve order
        - generators: a dict with the generators parameters for polynomial
                      interpolation: w, w_order, k1, k2, H, H1, H2
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

    generators = {'w': w, 'w_order': w_order, 'k1': k1,
                  'k2': k2, 'H': H, 'H1': H1, 'H2': H2, }

    # TODO: if w_order > num_gates since w_order is the smallest power of 2
    #       greater than num_gates, we need to pad the circuit
    a, b, c = build_permutations(H, H1, H2, circuit)

    QL_coeff = utils.interpolate(H, circuit['ql'], q)
    QR_coeff = utils.interpolate(H, circuit['qr'], q)
    QO_coeff = utils.interpolate(H, circuit['qo'], q)
    QM_coeff = utils.interpolate(H, circuit['qm'], q)
    QC_coeff = utils.interpolate(H, circuit['qc'], q)

    circuit =  {'QL_coeff': QL_coeff, 'QR_coeff': QR_coeff,
                'QO_coeff': QO_coeff, 'QM_coeff': QM_coeff,
                'QC_coeff': QC_coeff, 'a': a, 'b': b, 'c': c}

    return {'gp': gp, 'circuit': circuit, 'num_gates': n, 'order': q,
            'generators': generators}


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


def generate_proof(setup, public_inputs, trace):
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
        A proof object that can be verified by the verifier

    """
    transcript = str(setup) + str(public_inputs)

    q = setup['order']
    n = setup['num_gates']
    gp = setup['gp']  # the trusted setup parameters
    w, k1, k2, H, H1, H2 = [setup['generators'][x] for x in
        ['w', 'k1', 'k2', 'H', 'H1', 'H2']]

    # ROUND 1
    # Commit to the polynomials A, B, C representing the left, right
    # and output values of the gates, randominzed for zero-knowledge

    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11 = [
        secrets.randbelow(q) for _ in range(11)]

    x = sympy.symbols('x')

    # Zh is the polynomial that is zero on all the elements of H
    # Zh = w^n - 1 ; where n is the order of w
    coeff_Zh = [1] + [0] * (len(H) - 1) + [-1]
    Zh = Poly(coeff_Zh)

    left, right, output = [trace[x] for x in ['left', 'right', 'output']]

    A = utils.interpolate(H, left, q)
    B = utils.interpolate(H, right, q)
    C = utils.interpolate(H, output, q)

    A = Poly(b1 * x + b2) * Zh + Poly(A)
    B = Poly(b3 * x + b4) * Zh + Poly(B)
    C = Poly(b5 * x + b6) * Zh + Poly(C)

    com_A = kzg.commit(A, gp)
    com_B = kzg.commit(B, gp)
    com_C = kzg.commit(C, gp)

    transcript += str(com_A) + str(com_B) + str(com_C)

    # ROUND 2
    # Commit to polynomial Z, which eccodes the permutation constraints
    beta = utils.fiat_shamir([transcript, 0], q)
    gamma = utils.fiat_shamir([transcript, 1], q)

    a, b, c = [setup['circuit'][x] for x in ['a', 'b', 'c']]

    acc = [1]

    for i in range(1, n):
        num = sympy.Mod((left[i-1] + beta * H[i-1] + gamma)
               * (right[i-1] + beta * H1[i-1] + gamma)
               * (output[i-1] + beta * H2[i-1] + gamma), q)

        den = sympy.Mod((left[i-1] + beta * a[i-1] + gamma)
               * (right[i-1] + beta * b[i-1] + gamma)
               * (output[i-1] + beta * c[i-1] + gamma), q)

        acc.append(sympy.Mod(acc[i-1] * num * sympy.mod_inverse(den, q),  q))

    acc_coeff = utils.interpolate(H, acc, q)

    Z = (Poly([b7, b8, b9]) * Zh + Poly(acc_coeff))

    com_Z = kzg.commit(Z, gp)

    transcript += str(com_Z)

    # ROUND 3
    # Commit to polynomial T (split in three) that encodes most of the trace

    alfa = utils.fiat_shamir(transcript, q)

    PI = Poly(utils.interpolate(H, [-x for x in public_inputs], q))

    S1 = Poly(utils.interpolate(H, a, q))
    S2 = Poly(utils.interpolate(H, b, q))
    S3 = Poly(utils.interpolate(H, c, q))

    QL, QR, QM, QO, QC = [Poly(setup['circuit'][x]) for x in
        ['QL_coeff', 'QR_coeff', 'QM_coeff', 'QO_coeff', 'QC_coeff']]

    # Zw is Z(wX)
    Z_coeff = Z.all_coeffs()
    Zw = Poly([(pow(H[1], len(Z_coeff) - 1 - i, q) * coeff) % q
                for (i, coeff) in enumerate(Z_coeff)])

    # L1 is the Lagrange poly over H so that L1(1) = 1 and
    # L1 = 0 for all other roots of unity in H.
    # Get L1 by interpolating the vector (1, 0, 0, 0)
    L1 = Poly(utils.interpolate(H, [1] + [0] * (len(H)-1), q))

    T1 = A * B * QM + A * QL + B * QR + C * QO + PI + QC

    T2 = Poly(alfa * ((A + beta * x + gamma) * (B + beta * k1 * x + gamma)
                    * (C + beta * k2 * x + gamma) * Z))

    T3 = Poly(-1 * alfa * ((A + beta * S1 + gamma) * (B + beta * S2 + gamma)
                   * (C + beta * S3 + gamma) * Zw))

    T4 = Poly(alfa**2 * ((Z - 1) * L1))

    T_num = T1 + T2 + T3 + T4
    T, rem = T_num.div(Zh)
    assert rem == 0

    T_coeff = [x % q for x in T.all_coeffs()]

    Tlo_coeff = T_coeff[-n:]
    Tmid_coeff = T_coeff[-2*n:-n]
    Thi_coeff = T_coeff[:-2*n]

    Tlo = Poly(Tlo_coeff) + Poly(b10 * x**n)
    Tmid = Poly(Tmid_coeff) + Poly(-b10 + b11* x**n)
    Thi = Poly(Thi_coeff) + Poly([-b11])


    com_Tlo, com_Tmid, com_Thi = [
            kzg.commit(x, gp) for x in [Tlo, Tmid, Thi]]

    transcript += str(com_Tlo) + str(com_Tmid) + str(com_Thi)

    # ROUND 4

    zeta = utils.fiat_shamir(transcript, q)

    a_=  A.subs(x, zeta) % q
    b_=  B.subs(x, zeta) % q
    c_=  C.subs(x, zeta) % q
    s1_= S1.subs(x, zeta) % q
    s2_= S2.subs(x, zeta) % q
    zw_= Zw.subs(x, zeta) % q
    pi_ = PI.subs(x, zeta) % q
    l1_ = L1.subs(x, zeta) % q
    zh_ = Zh.subs(x, zeta) % q

    transcript += str(a_) + str(b_) + str(c_) + str(s1_) + str(s2_) + str(zw_)

    # ROUND 5

    v = utils.fiat_shamir(transcript, q)

    R = Poly(
            a_ * b_ * QM + a_ * QL + b_ * QR + c_ * QO + pi_ + QC
            + alfa * ((a_ + beta * zeta + gamma) * (b_ + beta * k1 * zeta + gamma)
                      * (c_ + beta * k2 * zeta + gamma) * Z)
            - alfa * ((a_ + beta * s1_ + gamma) * (b_ + beta * s2_ + gamma)
                      * (c_ + beta * S3 + gamma) * zw_)
            + alfa**2 * ((Z - 1) * l1_)
            - zh_ * (Tlo + zeta**n * Tmid + zeta**(2*n) * Thi)
    )

    Wzeta, rem = Poly(R + v * (A - a_) + v**2 * (B - b_) + v**3 * (C - c_)
                     + v**4 * (S1 - s1_) + v**5 * (S2 - s2_)).div(Poly(x - zeta))
    assert rem == 0

    Wzeta_w, rem = Poly(Z - zw_).div(Poly(x - zeta * w))
    assert rem == 0

    com_Wzeta, com_Wzeta_w = [kzg.commit(x, gp) for x in [Wzeta, Wzeta_w]]
    transcript += str(com_Wzeta) + str(com_Wzeta_w)

    proof = {'com_A': com_A, 'com_B': com_B, 'com_C': com_C, 'com_Z': com_Z,
             'com_Tlo': com_Tlo, 'com_Tmid': com_Tmid, 'com_Thi': com_Thi,
             'com_Wzeta': com_Wzeta, 'com_Wzeta_w': com_Wzeta_w, 'a_': a_,
             'b_': b_, 'c_': c_, 's1_': s1_, 's2_': s2_, 'zw_': zw_}

    return proof


def test():
    setup = run_setup('circuit_example')

    left = [3, 4, 5, 9]
    right = [3, 4, 5, 16]
    output = [9, 16, 25, 25]
    trace = {'left': left, 'right': right, 'output': output}

    public_inputs = []

    proof = generate_proof(setup, public_inputs, trace)
    print(proof)


if __name__ == '__main__':
    test()