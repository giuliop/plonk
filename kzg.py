"""
An implementation of the KZG polynomial commitment scheme.
We use the py_ecc library for the elliptic curve operations.
"""

import secrets
from py_ecc.optimized_bn128 import optimized_curve as curve, pairing
from sympy import Poly as sympy_Poly, symbols as sympy_symbols, FF as sympy_FF

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

def random_poly(d):
    """Generate a random polynomial of degree d.

    Args:
        d: The degree of the polynomial.

    Returns:
        A list of coefficients of the polynomial.
    """
    return [secrets.randbelow(curve.curve_order) for _ in range(d+1)]

def commit(poly, h):
    """Commit to a polynomial.

    Args:
        poly: The polynomial to commit to as a list of coefficients,
        starting from the highest degree.
        h: parameters of the trusted setup

    Returns:
        The commitment to the polynomial, a point in G1.
    """
    degree = len(poly) - 1
    com_f = curve.Z1 # Zero element of G1

    for i, pi in enumerate(poly):
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
    return sum(coeff * x**(degree - i)
               for i, coeff in enumerate(poly)) % curve.curve_order

def evaluate(poly, h, u):
    """Open a polynomial at a point.

    Args:
        poly: The polynomial to evaluate as a list of coefficients,
              starting with the highest degree term.
        h: parameters of the trusted setup
        u: The point to evaluate the polynomial at.

    Returns:
        A tuple (v = poly(u), proof) where proof is an element of G1
        to be used in the verify function.
    """
    v = poly_at(poly, u)

    # calculate the polynomial q(x) such that:
    # poly(x) - poly(u) = q(x) * (x - u)
    q_x = calculate_q_x(poly, u)

    proof = commit(q_x, h)
    return v, proof

def verify(com_f, u, v, proof, h):
    """Verify an evaluation proof.

    Args:
        com_f: The commitment to the polynomial.
        u: The point to evaluate the polynomial at.
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

def calculate_q_x(poly, u):
    """Calculate the polynomial q(x) such that:
       poly(x) - poly(u) = q(x) * (x - u).

    Args:
        poly: The polynomial to evaluate as a list of coefficients.
              The coefficients start with the highest degree term.
        u: The point to evaluate the polynomial at.

    Returns:
        A list of coefficients of the polynomial q(x),
        starting with the highest degree term.
    """
    x = sympy_symbols('x')
    domain = sympy_FF(curve.curve_order)
    # domain = None

    f_x = sympy_Poly(poly, x, domain=domain)
    f_u = f_x.subs(x, u)
    g_x = f_x - f_u

    q_x, remainder = g_x.div(sympy_Poly(x - u, x, domain=domain))
    assert remainder % curve.curve_order == 0

    return [x % curve.curve_order for x in q_x.all_coeffs()]

def test_kzg():
    """Test the KZG polynomial commitment scheme."""
    d = secrets.randbelow(100)
    h = trusted_setup(d)

    poly = random_poly(d)
    com_f = commit(poly, h)

    u = secrets.randbelow(curve.curve_order-1) + 1
    v, proof = evaluate(poly, h, u)

    assert verify(com_f, u, v, proof, h)
    print("It works!")

if __name__ == "__main__":
    test_kzg()
