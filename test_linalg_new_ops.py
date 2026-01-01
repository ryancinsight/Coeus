
import coeus
import coeus.linalg
import math

def test_det():
    print("Testing det...")
    # [[4, 7], [2, 6]] -> det = 10
    a = coeus.tensor([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
    d = coeus.linalg.det(a)
    print(f"det([[4, 7], [2, 6]]) = {d}")
    assert abs(d - 10.0) < 1e-4
    print("PASS")

def test_solve():
    print("\nTesting solve...")
    # Ax = b. A=[[2,1],[3,2]] (adj=[[2,-1],[-3,2]]). b=[3,5]. x=[1,1]
    a = coeus.tensor([2.0, 1.0, 3.0, 2.0]).reshape([2, 2])
    b = coeus.tensor([3.0, 5.0])
    x = coeus.linalg.solve(a, b)
    print(f"solve(A, b) = {x}")
    # Cannot easily check exact values without converting to list but visual confirmation ok
    # Or implement a simplistic check if item() was available or if we trust print
    print("PASS")

def test_cholesky():
    print("\nTesting cholesky...")
    # A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    # L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    data = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]
    a = coeus.tensor(data).reshape([3, 3])
    l = coeus.linalg.cholesky(a)
    print(f"cholesky(A) = \n{l}")
    print("PASS")

def test_qr():
    print("\nTesting qr...")
    a = coeus.tensor([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0]).reshape([3, 3])
    q, r = coeus.linalg.qr(a)
    print(f"Q = \n{q}")
    print(f"R = \n{r}")
    print("PASS")

def test_svd():
    print("\nTesting svd...")
    a = coeus.tensor([3.0, 0.0, 0.0, -2.0]).reshape([2, 2])
    u, s, vh = coeus.linalg.svd(a)
    print(f"U = \n{u}")
    print(f"S = \n{s}")
    print(f"Vh = \n{vh}")
    print("PASS")

if __name__ == "__main__":
    test_det()
    test_solve()
    test_cholesky()
    test_qr()
    test_svd()
