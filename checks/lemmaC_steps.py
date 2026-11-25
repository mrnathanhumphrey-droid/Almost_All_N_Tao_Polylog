# Check every inequality of the (D,N)-slice + Minkowski proof of C_L = O(L^2).
# g[x,d] = P(X_L = x, D = d);  h[x,d] = E[Y; X_L = x, D = d]  (Y real, Y = sum 3^{j-1} 2^{-a[1,j]})
# Recursion: X_L = 2^{-a1}(1 + 3 X'), Y_L = 2^{-a1}(1 + 3 Y'), D = a1 + D'.
import sys, math
import numpy as np

LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 12

def exact_C(L, AMAX=80):
    # full collision C_L (all d), as in phi_and_lemmaB.py
    P = np.array([1.0])
    for n in range(1, L + 1):
        q = 3 ** n
        Z = np.zeros(q); Z[(1 + 3 * np.arange(3 ** (n - 1))) % q] = P
        inv2 = pow(2, -1, q); x = np.arange(q); out = np.zeros(q); u = 1
        for a in range(1, AMAX + 1):
            u = (u * inv2) % q
            out[(u * x) % q] += Z * 2.0 ** (-a)
        P = out / out.sum()
    return 3 ** L * (P ** 2).sum()

def run(L, dmax):
    g = np.zeros((1, dmax + 1)); g[0, 0] = 1.0
    h = np.zeros((1, dmax + 1))
    for n in range(1, L + 1):
        q = 3 ** n
        inv2 = pow(2, -1, q)
        base = 1 + 3 * np.arange(3 ** (n - 1))
        G = np.zeros((q, dmax + 1)); H = np.zeros((q, dmax + 1))
        u = 1
        for a in range(1, dmax + 1):
            u = (u * inv2) % q
            idx = (u * base) % q            # injective map x' -> x
            w = 2.0 ** (-a)
            G[idx, a:] += w * g[:, :dmax + 1 - a]
            H[idx, a:] += w * w * (g[:, :dmax + 1 - a] + 3 * h[:, :dmax + 1 - a])
        g, h = G, H
    return g, h

print("L  dmax | max_(x,d) n(n-1)3^L/S | max_d LHS/RHS slice | C_L  Mink(d<=dmax)+tail  explicit  | sum_d sqrt(beta)")
for L in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1, LMAX + 1):
    dmax = 3 * L if L <= 11 else 30
    g, h = run(L, dmax)
    q = 3 ** L
    d = np.arange(dmax + 1)
    ratio = 0.0
    for dd in range(dmax + 1):
        nc = np.round(g[:, dd] * 2.0 ** dd)
        Sc = h[:, dd] * 4.0 ** dd
        m = nc >= 2
        if m.any():
            ratio = max(ratio, (nc[m] * (nc[m] - 1) * q / Sc[m]).max())
    # slice inequality: 3^L sum_x f_d^2 <= 2 Q_d + 2 E[Y;D=d]
    lhs = q * np.einsum('ij,ij->j', g, g)
    PD = g.sum(axis=0); EY = h.sum(axis=0)
    Qd = q * 2.0 ** (-d) * PD
    rhs = 2 * Qd + 2 * EY
    ok = PD > 0
    sl = (lhs[ok] / rhs[ok]).max()
    # Minkowski with the analytic tail for d > dmax:  beta_d <= (1+L/3) 3^L (2/3)^d
    tail = math.sqrt(1 + L / 3) * 3 ** (L / 2) * sum((2 / 3) ** (dd / 2) for dd in range(dmax + 1, dmax + 400))
    mink = (np.sqrt(lhs).sum()) ** 2; tailterm = math.sqrt(2) * tail
    C = exact_C(L)
    beta = Qd + EY
    sb = np.sqrt(beta).sum()
    explicit = 2 * (1 + L / 3) * (math.sqrt(2 * L + 1) + 4.45) ** 2
    # Chernoff tail check: beta_d <= (1+L/3) 3^L (2/3)^d for all computed d
    ch = (beta[ok] / ((1 + L / 3) * q * (2 / 3) ** d[ok])).max()
    # window check: sum_{d=L}^{3L} sqrt(beta) <= sqrt(2L+1) sqrt(1+L/3)
    win = np.sqrt(beta[L:min(3 * L, dmax) + 1]).sum() / (math.sqrt(2 * L + 1) * math.sqrt(1 + L / 3))
    print(f"{L:2d} {dmax:3d} | {ratio:.4f} | {sl:.4f} | {C:.4f} {mink:.3f}(+tail {tailterm:.2e}) {explicit:.1f} | {sb:.4f}  chern {ch:.3e} win {win:.3f}  sumQ {Qd.sum():.6f} sumEY {EY.sum():.6f} (L/3={L/3:.4f})")
    sys.stdout.flush()
