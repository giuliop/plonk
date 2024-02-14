"""
An implementation of the KZG polynomial commitment scheme,
using the py_ecc library for elliptic curve operations.

Batch proofs are implemented as per the Plonk paper, and the
multi-polynomials, multi-value case is generalized for an
arbitrary number of values.
"""

import hashlib
import secrets
import sympy

from py_ecc.optimized_bls12_381 import optimized_curve as curve, pairing

# the factorization of q-1, where q is the curve order for the curve we use
factorization = [2**32, 3, 11, 19, 10177, 125527, 859267, 906349, 906349,
                 2508409, 2529403, 52437899, 254760293, 254760293]

def trusted_setup(d):
    """Generate the trusted setup for a polynomial commitment scheme.

    Args:
        d: The maxumum degree of the polynomials to commit.

    Returns:
        A tuple with the general parameters of the trusted setup:
        1) a list of g1 * (r^i) for i in 0,d
        2) the list [g2, g2 * r]
        where r is a random number

    """
    g1 = curve.G1
    g2 = curve.G2
    q = curve.curve_order
    r = secrets.randbelow(q - 2) + 2

    return ([curve.multiply(g1, pow(r, i, q)) for i in range(d+1)],
            [g2, curve.multiply(g2, r)])


def commit(poly, h):
    """Commit to a polynomial.

    Args:
        poly: The polynomial to commit to as a list of coefficients,
        starting from the highest degree, or as a sympy Poly.
        h: parameters of the trusted setup

    Returns:
        The commitment to the polynomial, a point in G1.
    """
    if isinstance(poly, sympy.Poly):
        poly = [x % curve.curve_order for x in poly.all_coeffs()]

    degree = len(poly) - 1
    com_f = curve.Z1 # Zero element of G1

    for i, pi in enumerate(poly):
        pi = pi % curve.curve_order
        d = degree - i
        com_f = curve.add(com_f, curve.multiply(h[0][d], pi))

    return com_f

def poly_at(poly, x):
    """Evaluate a polynomial at a point x.

    Args:
        poly: The polynomial to evaluate as a list of coefficients,
              starting with the highest degree term.
        x: The point to evaluate the polynomial at.

    Returns:
        The value of the polynomial at x.
    """
    degree = len(poly) - 1
    p = curve.curve_order
    return sum(coeff * pow(x, degree - i, p)
               for i, coeff in enumerate(poly)) % p

def single_open(poly, u, h):
    """Open a polynomial at a point.

    Args:
        poly: The polynomial to open as a list of coefficients,
              starting with the highest degree term.
        u: The point to open the polynomial at.
        h: parameters of the trusted setup

    Returns:
        A tuple (v = poly(u), proof) where proof is an element of G1
        to be used in the verification function.
    """
    v = poly_at(poly, u)

    # calculate the polynomial q(x) such that:
    # poly(x) - poly(u) = q(x) * (x - u)
    q_x = calculate_q_x(poly, u)

    proof = commit(q_x, h)
    return v, proof

def batch_open(polys, u, r, h):
    """Open multiple polynomials at the same point.

    Args:
        polys: A list of polynomials to open.
        u: The point to open the polynomials at.
        r: A random parameter
        h: parameters of the trusted setup

    Returns:
        A tuple (vs, proof) where vs is a list of the evaluation of
        the polynomials at u, and proof is an element of G1 to be
        used in the verification function.
    """
    vs = [poly_at(poly, u) for poly in polys]

    # calculate the polynomial h(x) such that:
    # h(x) = sum( r^i * (polys[i](x) - polys[i](u)) / (x - u) )
    # for i in 0, len(polys)
    h_x = calculate_h_x(polys, u, r)

    proof = commit(h_x, h)

    return vs, proof

def batch_multi_open(polys_groups, us, rs, h):
    """Open multiple polynomials at different points.
       Each group of polynomials is opened at the same point.

    Args:
        polys_groups: A list of lists of polynomials to open.
        us: The points to open the polynomials at, each point
            corresponding to a group of polynomials.
        rs: A list of random parameters
        h: parameters of the trusted setup

    Returns:
        A tuple (vs_groups, proofs) where vs_groups is a list of lists
        of the evaluation of the polynomials at the corresponding points,
        and proofs is a list of elements of G1 to be used in the
        verification function.
    """
    vs_groups = []
    proofs = []
    for polys, u, r in zip(polys_groups, us, rs):
        vs, proof = batch_open(polys, u, r, h)
        vs_groups.append(vs)
        proofs.append(proof)

    return vs_groups, proofs

def single_verify(com_f, u, v, proof, h):
    """Verify an evaluation proof.

    Args:
        com_f: The commitment to the polynomial.
        u: The point to open the polynomial at.
        v: The value of the polynomial at u.
        proof: The proof of the evaluation.
        h: parameters of the trusted setup

    Returns:
        True if the proof is valid, False otherwise.
    """
    g1 = h[0][0]
    g2 = h[1][0]

    left = pairing(g2,
                   curve.add(com_f,
                             curve.multiply(curve.neg(g1), v)))

    right = pairing(curve.add(h[1][1],
                              curve.multiply(curve.neg(g2), u)),
                    proof)

    return left == right

def batch_verify(com_fs, u, vs, r, proof, h):
    """Verify a batch evaluation proof for multiple polys and
    the same evaluation point for all.

    Args:
        com_fs: A list of commitments to the polynomials.
        u: The point to open the polynomials at.
        vs: A list of the evaluation of the polynomials at u.
        r: A random parameter
        proof: The proof of the evaluation.
        h: parameters of the trusted setup

    Returns:
        True if the proof is valid, False otherwise.
    """
    g1 = h[0][0]
    g2 = h[1][0]
    p = curve.curve_order

    F = curve.Z1 # Zero element of G1
    for i, com_f in enumerate(com_fs):
        F = curve.add(F, curve.multiply(com_f, pow(r, i, p)))

    s = curve.multiply(g1, sum(v * pow(r, i, p) for i,v in enumerate(vs)))

    e1 = pairing(g2, curve.add(F, curve.neg(s)))

    e2 = pairing(curve.add(h[1][1],
                              curve.multiply(curve.neg(g2), u)),
                    curve.neg(proof))

    return e1 * e2 == curve.FQ12.one()

def batch_multi_verify(com_fs_groups, us, vs_groups, rs, proofs, h):
    """Verify a batch evaluation proof for multiple polys and
    multiple evaluation points.

    Args:
        com_fs_groups: A list of lists of  commitments to the polynomials.
        us: The points to open the polynomials at.
        vs_groups: A list of lists of the evaluation of the polynomials
                   at the corresponding points.
        rs: A list of random parameters
        proofs: A list of proofs of the evaluation.
        h: parameters of the trusted setup

    Returns:
        True if the proof is valid, False otherwise.
    """
    g1 = h[0][0]
    g2 = h[1][0]
    p = curve.curve_order

    # other random parameters
    _rs = [1]
    for _ in range(len(com_fs_groups) - 1):
        _rs.append(secrets.randbelow(p - 1) + 1)

    F = curve.Z1 # Zero element of G1
    for j, (_r, r) in enumerate(zip(_rs, rs)):
        f1 = curve.Z1
        f2 = 0
        for i, (com_f, v) in enumerate(zip(com_fs_groups[j], vs_groups[j])):
            f1 = curve.add(f1, curve.multiply(com_f, pow(r, i, p)))
            f2 += v * pow(r, i, p)
        f2 = curve.multiply(g1, f2)
        F = curve.add(F, curve.multiply(curve.add(f1, curve.neg(f2)), _r))

    e1_right = F
    e2_right = curve.Z1
    for proof, _r, u in zip(proofs, _rs, us):
        e1_right = curve.add(e1_right, curve.multiply(proof, (_r * u) % p))
        e2_right = curve.add(e2_right, curve.multiply(curve.neg(proof), _r))

    e1 = pairing(g2, e1_right)
    e2 = pairing(h[1][1], e2_right)
    return e1 * e2 == curve.FQ12.one()

def calculate_q_x(poly, u):
    """Calculate the polynomial q(x) such that:
       poly(x) - poly(u) = q(x) * (x - u).

    Args:
        poly: The polynomial poly(x) as a list of coefficients.
              The coefficients start with the highest degree term.
        u: The point to evaluate the polynomial at.

    Returns:
        A list of coefficients of the polynomial q(x),
        starting with the highest degree term.
    """
    x = sympy.symbols('x')
    domain = sympy.FF(curve.curve_order)

    f_x = sympy.Poly(poly, x, domain=domain)
    f_u = sympy.Poly(f_x.subs(x, u), x, domain=domain)
    g_x = f_x - f_u

    q_x, remainder = g_x.div(sympy.Poly(x - u, x, domain=domain))
    assert remainder % curve.curve_order == 0

    return [x % curve.curve_order for x in q_x.all_coeffs()]

def calculate_h_x(polys, u, r):
    """Calculate the polynomial h(x) such that:
       h(x) = sum( r^i * (polys[i](x) - polys[i](u)) / (x - u) )
       for i in 0, len(polys).

    Args:
        polys: A list of polynomials with coefficients as lists
               starting with the highest degree term.
        u: The point to evaluate the polynomials at.
        r: A random parameter

    Returns:
        A list of coefficients of the polynomial h(x),
        starting with the highest degree term.
    """
    x = sympy.symbols('x')
    domain = sympy.FF(curve.curve_order)

    h_x = sympy.Poly(0, x, domain=domain)
    for i, poly in enumerate(polys):
        f_x = sympy.Poly(poly, x, domain=domain)
        f_u = sympy.Poly(f_x.subs(x, u), x, domain=domain)
        g_x = f_x - f_u

        q_x, remainder = g_x.div(sympy.Poly(x - u, x, domain=domain))
        assert remainder % curve.curve_order == 0

        h_x = h_x + pow(r, i, curve.curve_order) * q_x

    return [x % curve.curve_order for x in h_x.all_coeffs()]

# --------------------------------------------------------------- #
# The following functions are used for testing the implementation #

def random_poly(d):
    """Generate a random polynomial of degree d.

    Args:
        d: The degree of the polynomial.

    Returns:
        A list of coefficients of the polynomial.
    """
    return [secrets.randbelow(curve.curve_order)
           for _ in range(d+1)]


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


def test_single_commit():
    """Test the KZG polynomial commitment scheme for one poly,
    one value to open."""
    d = secrets.randbelow(20)
    h = trusted_setup(d)

    poly = random_poly(d)
    com_f = commit(poly, h)

    # the point to open the polynomial at
    u = fiat_shamir([com_f, h], curve.curve_order)

    v, proof = single_open(poly, u, h)

    assert single_verify(com_f, u, v, proof, h)


def test_multi_poly_single_value_commit():
    """Test the KZG polynomial commitment scheme for multiple polys,
    same value for all polys to open."""
    d = secrets.randbelow(20)
    h = trusted_setup(d)

    polys_count = 10
    polys = [random_poly(d) for _ in range(polys_count)]
    com_fs = [commit(poly, h) for poly in polys]

    # the point to open the polynomials at
    u = fiat_shamir([com_fs, h], curve.curve_order)
    # a random parameter
    r = fiat_shamir(u, curve.curve_order)

    vs, proof = batch_open(polys, u, r, h)

    assert batch_verify(com_fs, u, vs, r, proof, h)


def test_multi_poly_multi_value_commit():
    """Test the KZG polynomial commitment scheme for multiple polys,
    multiple values to open."""
    d = secrets.randbelow(20)
    h = trusted_setup(d)

    values_count = 3
    polys_per_value_count = [2,4,3]
    # The idea is that we have 2 polys that evaluate to the first
    # value, 4 polys that evaluate to the second value and so on.

    polys_groups = []
    for n in polys_per_value_count:
        polys_groups.append([random_poly(d) for _ in range(n)])

    com_fs_groups = []
    for polys in polys_groups:
        com_fs_groups.append([commit(poly, h) for poly in polys])

    # the points to open the polynomials at
    us = []
    to_hash = [com_fs_groups, h]
    for _ in range(values_count):
        to_hash = fiat_shamir(to_hash, curve.curve_order)
        us.append(to_hash)

    # random parameters
    rs = []
    for _ in range(values_count):
        to_hash = fiat_shamir(to_hash, curve.curve_order)
        rs.append(to_hash)

    vs_groups, proofs = batch_multi_open(polys_groups, us, rs, h)

    assert batch_multi_verify(com_fs_groups, us,
                              vs_groups, rs, proofs, h)


if __name__ == "__main__":
    test_single_commit()
    test_multi_poly_single_value_commit()
    test_multi_poly_multi_value_commit()
    print("All tests passed!")
