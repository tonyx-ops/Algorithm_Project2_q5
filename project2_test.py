# test_dnc_convex_hull.py
# PyTest unit tests for dnc_convex_hull.py

import math
import random
import itertools
import numpy as np
import pytest

# Import the function under test
import project2_q5 as ch

# ---------- Helpers (for tests only) ----------

EPS = 1e-9

def ref_orient(points):
    """
    Reference convex hull (CCW) using cross product.
    Drops interior collinear boundary points (<= tests),
    matching the behavior of your implementation.
    """
    P = sorted(set(points))
    if len(P) <= 1:
        return P[:]

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower = []
    for p in P:
        while len(lower) >= 2 and orient(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(P):
        while len(upper) >= 2 and orient(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def is_ccw(poly):
    """Area > 0 for CCW polygon; tolerates degenerate cases."""
    if len(poly) < 3:
        return True
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        area2 += x1 * y2 - y1 * x2
    return area2 >= -EPS

def is_convex_ccw(poly):
    """Check all consecutive turns are non-negative (CCW or collinear)."""
    m = len(poly)
    if m < 3:
        return True
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    for i in range(m):
        a, b, c = poly[i], poly[(i+1) % m], poly[(i+2) % m]
        if orient(a, b, c) < -EPS:
            return False
    return True

def all_points_inside_or_on(poly, pts):
    """For CCW polygon, every point p should satisfy orient(vi, v(i+1), p) >= 0."""
    if not poly:
        return len(pts) == 0
    if len(poly) == 1:
        # Single vertex hull: all points must coincide with that vertex to be "inside"
        vx, vy = poly[0]
        return all(abs(px - vx) <= EPS and abs(py - vy) <= EPS for (px, py) in pts)
    if len(poly) == 2:
        # Segment hull: points must lie on or between the segment endpoints (collinear and within bounds)
        (x1, y1), (x2, y2) = poly
        dx, dy = x2 - x1, y2 - y1
        L2 = dx*dx + dy*dy
        if L2 <= EPS:
            return all(abs(px - x1) <= EPS and abs(py - y1) <= EPS for (px, py) in pts)
        for (px, py) in pts:
            # collinearity
            if abs((px - x1) * dy - (py - y1) * dx) > 1e-7:
                return False
            # within bounding box
            if (min(x1, x2) - 1e-7 <= px <= max(x1, x2) + 1e-7 and
                min(y1, y2) - 1e-7 <= py <= max(y1, y2) + 1e-7):
                continue
            return False
        return True
    # Proper polygon
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    m = len(poly)
    for p in pts:
        for i in range(m):
            a, b = poly[i], poly[(i+1) % m]
            if orient(a, b, p) < -1e-7:
                return False
    return True

def as_tuples(arr):
    """Convert Nx2 numpy array to list of tuples."""
    return [tuple(p) for p in np.asarray(arr)]

# ---------- Unit tests ----------

def test_empty():
    assert ch.convex_hull_divide_and_conquer([]) == []

def test_single_point():
    P = [(2.0, 3.0)]
    assert ch.convex_hull_divide_and_conquer(P) == P

def test_two_points():
    P = [(0.0, 0.0), (1.0, 1.0)]
    H = ch.convex_hull_divide_and_conquer(P)
    assert len(H) == 2
    assert set(H) == set(P)

def test_three_points_triangle():
    P = [(0,0), (2,0), (1,1)]
    H = ch.convex_hull_divide_and_conquer(P)
    # Triangle hull should be all 3 in CCW order (rotation allowed)
    assert len(H) == 3
    assert set(H) == set(P)
    assert is_ccw(H) and is_convex_ccw(H)

def test_square():
    P = [(0,0), (1,0), (1,1), (0,1)]
    H = ch.convex_hull_divide_and_conquer(P)
    assert len(H) == 4
    assert set(H) == set(P)
    assert is_ccw(H) and is_convex_ccw(H)

def test_collinear_many():
    # All points collinear; hull should be the two endpoints only.
    P = [(i, 2*i) for i in range(-5, 6)]
    H = ch.convex_hull_divide_and_conquer(P)
    assert len(H) == 2
    assert set(H) == {(-5, -10), (5, 10)}

def test_with_duplicates():
    P = [(0,0), (1,0), (1,1), (0,1), (1,0), (0,0), (0,1)]
    H = ch.convex_hull_divide_and_conquer(P)
    assert len(H) == 4
    assert set(H) == {(0,0), (1,0), (1,1), (0,1)}

@pytest.mark.parametrize("n,seed", [(50, 1), (200, 2), (1000, 3)])
def test_matches_reference_monotone_chain(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2)) * 100.0
    P = as_tuples(pts)

    H1 = ch.convex_hull_divide_and_conquer(P)
    H2 = ref_orient(P)

    # Same vertex set (order may differ but both are CCW by construction)
    assert set(H1) == set(H2)
    assert is_ccw(H1) and is_convex_ccw(H1)

def test_all_input_points_inside():
    rng = np.random.default_rng(42)
    pts = (rng.random((500, 2)) * 100.0).tolist()
    P = [tuple(p) for p in pts]
    H = ch.convex_hull_divide_and_conquer(P)

    # Hull should be convex CCW and contain all points
    assert is_ccw(H) and is_convex_ccw(H)
    assert all_points_inside_or_on(H, P)

def test_idempotence_hull_of_hull():
    # hull(hull(P)) should equal hull(P)
    rng = np.random.default_rng(7)
    P = as_tuples(rng.random((300, 2)) * 10.0)
    H = ch.convex_hull_divide_and_conquer(P)
    H2 = ch.convex_hull_divide_and_conquer(H)
    assert H2 == H

def test_rotation_invariance_square():
    # Rotating a square should not change vertex set of hull
    square = [(0,0), (1,0), (1,1), (0,1)]
    theta = math.radians(30)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta),  math.cos(theta)]])
    sq_rot = [tuple((R @ np.array(p)).tolist()) for p in square]
    H1 = ch.convex_hull_divide_and_conquer(square)
    H2 = ch.convex_hull_divide_and_conquer(sq_rot)
    # compare sets with small rounding, since rotation introduces floats
    def round2(P, k=6): return { (round(x, k), round(y, k)) for (x,y) in P }
    assert round2(H1) == round2(ref_orient(square))
    assert round2(H2) == round2(ref_orient(sq_rot))

@pytest.mark.parametrize("n", [1, 2, 3, 10, 100])
def test_no_duplicate_vertices_in_output(n):
    rng = np.random.default_rng(123 + n)
    P = as_tuples(rng.integers(-5, 6, size=(n, 2)))
    H = ch.convex_hull_divide_and_conquer(P)
    assert len(H) == len(set(H)), "Hull output contains duplicate vertices"
