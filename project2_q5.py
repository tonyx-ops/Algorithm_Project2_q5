"""
Module: experimental_vs_theory_project2
Description: Empirically evaluate the divide-and-conquer convex hull algorithm.
             - Generate random 2D point sets.
             - Measure runtime of the D&C hull implementation.
             - Compare against theoretical O(n log n) growth (normalized n log n curve).
             - Produce plots: points-only, points+convex hull, and runtime vs theory.

@author: Juntao (Tony) Xue
@date: October 15, 2025
@version: 2.0
"""

__author__  = "Juntao (Tony) Xue"
__version__ = "2.0"
__date__    = "2025-10-15"
__project__ = "Project 2: Divide-and-Conquer Convex Hull — Experimental vs. O(n log n)"


# dnc_convex_hull.py
from typing import List, Tuple
from statistics import median
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import os


Point = Tuple[float, float]

# ---------- Geometry helpers ----------
def orient(a: Point, b: Point, c: Point) -> float:
    """
    area (2x) of triangle abc = cross((b-a), (c-a)).
    cross product to determine clockwise rotation
    > 0  => a->b->c is a left turn (counterclockwise)
    < 0  => right turn (clockwise)
    == 0 => collinear
    """
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def next(i: int, m: int) -> int:
    """Circular 'next' index on a polygon of size m (Counterclockwise order)."""
    return (i + 1) % m

def prev(i: int, m: int) -> int:
    """Circular 'previous' index on a polygon of size m (Counterclockwise order)."""
    return (i - 1 + m) % m

def rightmost_index(H: List[Point]) -> int:
    """
    Index of the rightmost vertex in hull H.
    compare x value, if ties, then choose highest y.
    """
    mx, my, idx = H[0][0], H[0][1], 0
    for i, (x, y) in enumerate(H):
        if x > mx or (x == mx and y > my):
            mx, my, idx = x, y, i
    return idx

def leftmost_index(H: List[Point]) -> int:
    """
    Index of the leftmost vertex in hull H.
    compare x value, if ties, then choose highest y.
    """
    mnx, mny, idx = H[0][0], H[0][1], 0
    for i, (x, y) in enumerate(H):
        if x < mnx or (x == mnx and y < mny):
            mnx, mny, idx = x, y, i
    return idx

# ---------- Small hull (≤3) with tiny monotone chain ----------
def hull_of_small(S: List[Point]) -> List[Point]:
    """
    Convex hull for |S| <= 3.
    We use a mini monotone-chain pass that:
      - sorts and deduplicates points,
      - builds lower and upper chains,
      - uses <= in the orientation tests to DROP interior collinear points.
    Returns vertices in CCW order.
    """
    S = sorted(set(S))  # delete duplicate + sort by (x,y)
    if len(S) <= 1:
        return S[:]

    # Build lower chain
    L: List[Point] = []
    for p in S:
        # While last turn is not strictly left, pop the last vertex
        while len(L) >= 2 and orient(L[-2], L[-1], p) <= 0:
            L.pop()
        L.append(p)

    # Build upper chain
    U: List[Point] = []
    for p in reversed(S):
        while len(U) >= 2 and orient(U[-2], U[-1], p) <= 0:
            U.pop()
        U.append(p)

    # Concatenate without duplicating the first/last point
    return L[:-1] + U[:-1]

# ---------- Tangents  ----------
def find_upper_tangent(HL: List[Point], HR: List[Point]) -> Tuple[int, int]:
    """
    Find indices (i, j) such that the line HL[i]--HR[j] is the common UPPER tangent:
      - all vertices of HL and HR lie on or BELOW this line.
    We start from a good guess (rightmost of HL, leftmost of HR)
    and rotate the line until both sides are supporting.
    """
    i = rightmost_index(HL)
    j = leftmost_index(HR)
    mL, mR = len(HL), len(HR)

    changed = True
    while changed:
        changed = False
        # Rotate around HL clockwise while the line is not a supporting line for HL
        while orient(HR[j], HL[i], HL[prev(i, mL)]) >= 0:
            i = prev(i, mL)
            changed = True
        # Rotate around HR counterclockwise while the line is not a supporting line for HR
        while orient(HL[i], HR[j], HR[next(j, mR)]) <= 0:
            j = next(j, mR)
            changed = True
    return i, j

def find_lower_tangent(HL: List[Point], HR: List[Point]) -> Tuple[int, int]:
    """
    Find indices (i, j) such that the line HL[i]--HR[j] is the common LOWER tangent:
      - all vertices of HL and HR lie on or ABOVE this line.
    Symmetric to the upper tangent, but we rotate in the opposite directions.
    """
    i = rightmost_index(HL)
    j = leftmost_index(HR)
    mL, mR = len(HL), len(HR)

    changed = True
    while changed:
        changed = False
        # Rotate around HL counterclockwise
        while orient(HR[j], HL[i], HL[next(i, mL)]) <= 0:
            i = next(i, mL)
            changed = True
        # Rotate around HR clockwise
        while orient(HL[i], HR[j], HR[prev(j, mR)]) >= 0:
            j = prev(j, mR)
            changed = True
    return i, j

def splice(HL: List[Point], HR: List[Point],
           iu: int, ju: int, il: int, jl: int) -> List[Point]:
    """
    Splice the OUTER arcs between the tangent endpoints to form the merged hull (CCW).
      - On HR: take the CCW arc from ju to jl (inclusive).
      - On HL: take the CCW arc from il to iu (inclusive).
    Concatenate the two arcs. If the first and last vertices coincide, delete the duplicate.
    """
    mL, mR = len(HL), len(HR)

    # HR arc: ju -> jl (CCW on HR)
    R_chain: List[Point] = [HR[ju]]
    k = ju
    while k != jl:
        k = next(k, mR)
        R_chain.append(HR[k])

    # HL arc: il -> iu (CCW on HL)
    L_chain: List[Point] = [HL[il]]
    k = il
    while k != iu:
        k = next(k, mL)
        L_chain.append(HL[k])

    # Concatenate arcs to get merged CCW hull
    merged = R_chain + L_chain

    # Remove duplicate if we wrapped to the same point
    if len(merged) >= 2 and merged[0] == merged[-1]:
        merged.pop()
    return merged

def hull_rec(sorted_pts: List[Point]) -> List[Point]:
    """
    Recursive divide-and-conquer hull on points sorted by (x,y).
      - Divide: split into halves
      - Conquer: recursive hull on each half (returns CCW hulls)
      - Combine: find upper/lower tangents and splice outer arcs
    """
    n = len(sorted_pts)
    if n <= 3:  # Base case: handle with small monotone-chain
        return hull_of_small(sorted_pts)

    mid = n // 2
    HL = hull_rec(sorted_pts[:mid])   # left half hull (CCW)
    HR = hull_rec(sorted_pts[mid:])   # right half hull (CCW)

    # Merge via upper and lower tangents
    iu, ju = find_upper_tangent(HL, HR)
    il, jl = find_lower_tangent(HL, HR)
    return splice(HL, HR, iu, ju, il, jl)

def convex_hull_divide_and_conquer(P: List[Point]) -> List[Point]:
    """
    Entry point:
      - Sort and deduplicate points once (O(n log n)).
      - Recurse and merge hulls in linear time per level.
    Returns a CCW list of hull vertices with interior collinear points removed.
    """
    if not P:
        return []
    A = sorted(set(P))  # sort once + delete duplication by (x,y)
    if len(A) <= 3:
        return hull_of_small(A)
    return hull_rec(A)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)

    # --- change n to test different sizes, even n = 0/1/2 ---
    n = 1000
    pts = rng.random((n, 2)) * 10

    # -------- Figure 1: original points only --------
    plt.figure()
    if len(pts) > 0:
        plt.scatter(pts[:, 0], pts[:, 1], s=15, label="Points")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Original Points (n = {len(pts)})")
    plt.xlabel("x")
    plt.ylabel("y")
    if len(pts) > 0:
        plt.legend()

    # -------- Compute convex hull (divide & conquer) --------
    pts_list = [tuple(p) for p in pts]
    hull_list = convex_hull_divide_and_conquer(pts_list)  # your D&C function
    hull = np.array(hull_list, dtype=float) if hull_list else np.empty((0, 2))

    # -------- Figure 2: points + convex hull (robust for 0/1/2/≥3) --------
    plt.figure()
    if len(pts) > 0:
        plt.scatter(pts[:, 0], pts[:, 1], s=15, label="Points")

    if len(hull) == 1:
        plt.scatter(hull[0, 0], hull[0, 1], s=40, marker="x", label="Hull point")
    elif len(hull) == 2:
        plt.plot(hull[:, 0], hull[:, 1], "-", label="Hull edge")
    elif len(hull) >= 3:
        closed = np.vstack([hull, hull[0]])  # close the loop
        plt.plot(closed[:, 0], closed[:, 1], "-", label=f"Hull (|V|={len(hull)})")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Points + Convex Hull (|V| = {len(hull)})")
    plt.xlabel("x")
    plt.ylabel("y")
    if len(pts) > 0 or len(hull) > 0:
        plt.legend()
    plt.show()

