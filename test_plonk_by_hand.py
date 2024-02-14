"""Let's replicate the wonderful plonk example in the series:
   https://research.metastate.dev/plonk-by-hand-part-1/
   https://research.metastate.dev/plonk-by-hand-part-2-the-proof/
   https://research.metastate.dev/plonk-by-hand-part-3-verification/

   The plonk paper can be found at:
   https://eprint.iacr.org/2019/953.pdf
"""

import sympy

import circuit_compiler as cc
import plonk
import utils

q = 17  # prime field

def Poly(expr):
    """Convenience function to create sympy polynomials in the variable
       x in the domain of the field_order q (global variable).

    Args:
        expr: a list of coefficients starting from highest degree or
              a sympy expression with the symbol x (e.g., x**2 - 1)
    Returns:
        A sympy Poly built from expr
    """
    domain = sympy.FF(q)
    if isinstance(expr, list):
        return sympy.Poly(expr, sympy.symbols('x'), domain=domain)
    return sympy.Poly(expr, domain=domain)

def test_plonk_by_hand():
    """We replicate the entire thing using `assert` to verify we are
       getting the correct results as per the tutorial.
       We don't replicate the elliptic curve operations though, we
       just copy the results of those."""

    ##########                SETUP                ##########
    circuit = cc.parse_file('circuit_example')

    w = 4   # generator of order 4 in p
    k1, k2 = 2, 3  # to generate the cosets from the generator w
    H = [pow(w, i, q) for i in range(4)]
    H1 = [k1 * x % q for x in H]
    H2 = [k2 * x % q for x in H]

    assert H == [1, 4, 16, 13]
    assert H1 == [2, 8, 15, 9]
    assert H2 == [3, 12, 14, 5]

    a, b, c = plonk.build_permutations(H, H1, H2, circuit)

    QL_coeff = utils.interpolate(H, circuit['ql'], q)
    QR_coeff = utils.interpolate(H, circuit['qr'], q)
    QO_coeff = utils.interpolate(H, circuit['qo'], q)
    QM_coeff = utils.interpolate(H, circuit['qm'], q)
    QC_coeff = utils.interpolate(H, circuit['qc'], q)

    n = circuit['num_gates']

    assert QL_coeff == [16, 4, 1, 13]
    assert QR_coeff == [16, 4, 1, 13]
    assert QO_coeff == [16]
    assert QM_coeff == [1, 13, 16, 5]
    assert QC_coeff == [0]
    assert a == [2, 8, 15, 3]
    assert b == [1, 4, 16, 12]
    assert c == [13, 9, 5, 14]

    ##########               PROVER                ##########

    # ROUND 1
    left = [3, 4, 5, 9]
    right = [3, 4, 5, 16]
    output = [9, 16, 25, 25]

    A_coeff = utils.interpolate(H, left, q)
    B_coeff = utils.interpolate(H, right, q)
    C_coeff = utils.interpolate(H, output, q)

    assert A_coeff == [3, 3, 13, 1]
    assert B_coeff == [13, 14, 3, 7]
    assert C_coeff == [4, 11, 5, 6]

    x = sympy.symbols('x')

    coeff_Zh = [1] + [0] * (len(H) - 1) + [-1]
    Zh = Poly(coeff_Zh)
    assert Zh == Poly(x**4 - 1)

    randoms = [7, 4, 11, 12, 16, 2]

    A = Poly(randoms[0] * x + randoms[1]) * Zh + Poly(A_coeff)
    B = Poly(randoms[2] * x + randoms[3]) * Zh + Poly(B_coeff)
    C = Poly(randoms[4] * x + randoms[5]) * Zh + Poly(C_coeff)

    assert [x % q for x in A.all_coeffs()] == [7, 4, 3, 3, 6, 14]
    assert [x % q for x in B.all_coeffs()] == [11, 12, 13, 14, 9, 12]
    assert [x % q for x in C.all_coeffs()] == [16, 2, 4, 11, 6, 4]

    com_A = (91, 66)
    com_B = (26, 45)
    com_C = (91, 35)

    # ROUND 2
    beta = 12   # random challenge from validator
    gamma = 13  # random challenge from validator
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

    acc_coeff = utils.interpolate(H, acc, q)
    assert acc_coeff == [14, 5, 16, 0]

    randoms += [14, 11, 7]

    Z = Poly([randoms[6], randoms[7], randoms[8]]) * Zh + Poly(acc_coeff)
    assert [x % q for x in Z.all_coeffs()] == [14, 11, 7, 14, 8, 5, 10]

    com_Z = (32, 59)

    # ROUND 3

    PI = Poly([0])    # no public inputs
    S1 = Poly(utils.interpolate(H, a, q))
    S2 = Poly(utils.interpolate(H, b, q))
    S3 = Poly(utils.interpolate(H, c, q))
    alfa = 15 # random challenge from validator

    # Zw is Z(wX)
    Z_coeff = Z.all_coeffs()
    Zw_coeff = [(pow(H[1], len(Z_coeff) - 1 - i, q) * coeff) % q
                for (i, coeff) in enumerate(Z_coeff)]
    Zw = Poly(Zw_coeff)
    assert [x % q for x in Zw_coeff] == [3, 10, 7, 12, 9, 3, 10]

    # L1 is the Lagrange poly over H so that L1(1) = 1 and
    # L1 = 0 for all other roots of unity in H.
    # Get L1 by interpolating the vector (1, 0, 0, 0)
    L1 = Poly(utils.interpolate(H, [1] + [0] * (len(H)-1), q))

    QM = Poly(QM_coeff)
    QL = Poly(QL_coeff)
    QR = Poly(QR_coeff)
    QO = Poly(QO_coeff)
    QC = Poly(QC_coeff)

    T1 = A * B * QM + A * QL + B * QR + C * QO + PI + QC

    T2 = Poly(alfa * ((A + beta * x + gamma) * (B + beta * k1 * x + gamma)
                    * (C + beta * k2 * x + gamma) * Z))

    T3 = Poly(-1 * alfa * ((A + beta * S1 + gamma) * (B + beta * S2 + gamma)
                   * (C + beta * S3 + gamma) * Zw))

    T4 = Poly(alfa**2 * ((Z - 1) * L1))

    T = T1 + T2 + T3 + T4

    assert [x % q for x in T1.all_coeffs()] == [9, 7, 8, 9, 5, 10, 5, 3, 3, 16, 4, 5, 0, 1]
    assert [x % q for x in T2.all_coeffs()] == [14, 14, 2, 15, 8, 16, 16, 4, 12, 1, 12, 12, 7, 8, 13, 10, 0, 3, 12, 1, 1, 0]
    assert [x % q for x in T3.all_coeffs()] == [14, 10, 0, 1, 4, 9, 0, 7, 9, 12, 4, 16, 2, 7, 2, 9, 11, 9, 7, 10, 3, 13]
    assert [x % q for x in T4.all_coeffs()] == [14, 8, 15, 12, 6, 0, 2, 5, 14, 9]
    assert [x % q for x in T.all_coeffs()] == [11, 7, 2, 16, 12, 8, 16, 11, 13, 3, 7, 3, 11, 16, 1, 0, 3, 11, 8, 4, 1, 6]

    T, rem = T.div(Zh)
    assert rem == 0

    assert [x % q for x in T.all_coeffs()] == [11, 7, 2, 16, 6, 15, 1, 10, 2, 1, 8, 13, 13, 0, 9, 13, 16, 11]

    # T has degree 3n+5; in the paper we split in three polys of degree
    # n-1, n-1, n+5 but here we do three polys of degree n+1 each
    T_coeff = [x % q for x in T.all_coeffs()]
    Thi_coeff = T_coeff[:n+2]
    Tmid_coeff = T_coeff[n+2:2*n+4]
    Tlo_coeff = T_coeff[2*n+4:]

    assert Thi_coeff == [11, 7, 2, 16, 6, 15]
    assert Tmid_coeff == [1, 10, 2, 1, 8, 13]
    assert Tlo_coeff == [13, 0, 9, 13, 16, 11]

    # The paper here randomized the coefficients of Tlo, Tmid and Thi
    # but the tutorial does not

    com_Tlo = (12,32)
    com_Tmid = (26, 45)
    com_Thi = (91, 66)

    # ROUND 4
    # from here on the tutorial differs from the paper, we follow the tutorial
    # pointing out the differences

    zeta = 5    # random challenge from the verifier

    # we do modulo q because sympy often returns negative numbers
    # which would make the assert below fail
    a_=  A.subs(x, zeta) % q
    b_=  B.subs(x, zeta) % q
    c_=  C.subs(x, zeta) % q
    s1_= S1.subs(x, zeta) % q
    s2_= S2.subs(x, zeta) % q
    zw_= Zw.subs(x, zeta)%  q
    # this one is not present in the paper
    t_= T.subs(x, zeta) % q

    assert a_== 15
    assert b_== 13
    assert c_== 5
    assert s1_== 1
    assert s2_== 12
    assert zw_== 15
    assert t_== 1

    # zero in the tutorial but otherwise necessary
    pi_ = PI.subs(x, zeta) % q

    l1_ = L1.subs(x, zeta) % q
    zh_ = Zh.subs(x, zeta) % q
    Tlo = Poly(Tlo_coeff)
    Tmid = Poly(Tmid_coeff)
    Thi = Poly(Thi_coeff)

    R = Poly(
            a_ * b_ * QM + a_ * QL + b_ * QR + c_ * QO + pi_ + QC
            + alfa * ((a_ + beta * zeta + gamma) * (b_ + beta * k1 * zeta + gamma)
                      * (c_ + beta * k2 * zeta + gamma) * Z)
            - alfa * ((a_ + beta * s1_ + gamma) * (b_ + beta * s2_ + gamma)
                      * (c_ + beta * S3 + gamma) * zw_)
            + alfa**2 * ((Z - 0) * l1_)
        # this line above in the paper is instead the following two lines:
            # + alfa**2 * ((Z - 1) * l1_)
            # - zh_ * (Tlo + zeta**n * Tmid + zeta**(2*n) * Thi)
    )
    r_ = R.subs(x, zeta) % q

    assert [x % q for x in R.all_coeffs()] == [16, 15, 8, 13, 9, 16, 0]
    assert r_ == 15

    # ROUND 5

    v = 12   # random challenge from the verifier

    Wzeta_num = Poly(Tlo + zeta**(n+2) * Tmid + zeta**(2*n+4) * Thi - t_
                   + v * (R - r_) + v**2 * (A - a_) + v**3 * (B - b_)
                   + v**4 * (C - c_) + v**5 * (S1 - s1_) + v**6 * (S2 - s2_)
                   )
    assert [x % q for x in Wzeta_num.all_coeffs()] == [5, 12, 11, 8, 3, 2, 5]

    Wzeta, rem = Wzeta_num.div(Poly(x - zeta))
    assert rem == 0

    assert [x % q for x in Wzeta.all_coeffs()] == [5, 3, 9, 2, 13, 16]

    # in the paper we have the following instead:
    # Wzeta, rem = Poly(R + v * (A - a_) + v**2 * (B - b_) + v**3 * (C - c_)
                      # + v**4 * (S1 - s1_) + v**5 * (S2 - s2_)).div(Poly(x - zeta))

    Wzeta_w, rem = Poly(Z - zw_).div(Poly(x - zeta * w))
    assert rem == 0

    assert [x % q for x in Wzeta_w.all_coeffs()] == [14, 2, 13, 2, 14, 13]

    com_Wzeta = (91, 35)
    com_Wzeta_w = (65, 98)

    # We are done here is the proof! Note that the paper does not include r_
    proof = (com_A, com_B, com_C, com_Z, com_Tlo, com_Tmid, com_Thi, com_Wzeta,
             com_Wzeta_w, a_, b_, c_, s1_, s2_, r_, zw_)

    ##########             VERIFIER                ##########

    u = 4   # random challenge from the verifier

def print_poly(p):
    """Take a Poly expression and print its coefficients starting
    from the lowest degree. This is useful to compare with the tutorial."""
    p = Poly(p)
    p = p.all_coeffs()
    p.reverse()
    print([x % q for x in p])

if __name__ == '__main__':
    test_plonk_by_hand()