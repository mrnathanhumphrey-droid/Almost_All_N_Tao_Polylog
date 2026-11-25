# Phi(n) table and monotonicity (Lemma 3.2, Table 3); compatibility (Result 3.3); 2 is a primitive root mod 3^L;
# inequality (4) of Lemma B with the weaker constant 0.846/sqrt(n-L), and its unweighted form (Section 9.2).
import numpy as np
from fractions import Fraction
NMAX = 13
AMAX = 64
# distributions P[n] on Z/3^n, X_n = 2^{-a}(1 + 3 X_{n-1}), X_0 = 0
P = {0: np.array([1.0])}
for n in range(1, NMAX + 1):
    q = 3 ** n
    Z = np.zeros(q)
    prev = P[n - 1]
    idx = (1 + 3 * np.arange(3 ** (n - 1))) % q
    np.add.at(Z, idx, prev)
    inv2 = pow(2, -1, q)
    x = np.arange(q)
    out = np.zeros(q)
    u = 1
    for a in range(1, AMAX + 1):
        u = (u * inv2) % q
        np.add.at(out, (u * x) % q, Z * 2.0 ** (-a))
    out /= out.sum()
    P[n] = out

# sanity vs Tao line 617
print("P2*63:", np.round(P[2] * 63, 6))

# brute-force check of definition for n=4 using exact enumeration (a_i<=40)
def brute(n, amax=30):
    q = 3 ** n
    dist = {0: 1.0}  # state: (partial sum mod q, 2^{-a[1,j]} mod q, 3^{j}) folded
    states = {(0, 1): 1.0}
    inv2 = pow(2, -1, q)
    for j in range(1, n + 1):
        ns = {}
        for (s, w), p in states.items():
            ww = w
            for a in range(1, amax + 1):
                ww = (ww * inv2) % q
                key = ((s + pow(3, j - 1) * ww) % q, ww)
                ns[key] = ns.get(key, 0) + p * 2.0 ** (-a)
        states = ns
    out = np.zeros(q)
    for (s, w), p in states.items():
        out[s] += p
    return out / out.sum()
for n in (2, 3, 4):
    print("brute vs recursion n=%d maxdiff %.2e" % (n, np.abs(brute(n) - P[n]).max()))

phi = {}
for n in range(1, NMAX + 1):
    phi[n] = np.fft.fft(P[n])  # phi[n][xi] = sum_x P(x) e(-xi x/3^n)
units = {n: np.array([x for x in range(3 ** n) if x % 3]) for n in range(1, NMAX + 1)}
Phi = {n: np.abs(phi[n][units[n]]).max() for n in range(1, NMAX + 1)}
I = {n: (np.abs(phi[n][units[n]]) ** 2).sum() for n in range(1, NMAX + 1)}
coll = {n: 3 ** n * (P[n] ** 2).sum() for n in range(0, NMAX + 1)}
claimed = [0.378, 0.252, 0.177, 0.129, 0.096, 0.076, 0.061, 0.048, 0.038, 0.032, 0.026, 0.022]
print("n  Phi(n)   claimed  I_n   growth-law-diff  3^n*sumP^2")
for n in range(1, NMAX + 1):
    c = claimed[n - 2] if n >= 2 else float('nan')
    print(n, "%.5f" % Phi[n], c, "%.5f" % I[n], "%.5f" % (coll[n] - coll[n - 1]), "%.4f" % coll[n])

# compatibility (1.22): P_n mod 3^{n-1} == P_{n-1}
for n in range(2, NMAX + 1):
    red = P[n].reshape(3, 3 ** (n - 1)).sum(0)
    assert np.abs(red - P[n - 1]).max() < 1e-12
print("compatibility mod 3^{n-1}: ok")
# 2 primitive root mod 3^L
for L in range(1, NMAX + 1):
    q = 3 ** L; M = 2 * 3 ** (L - 1)
    assert pow(2, M // 2, q) != 1 and pow(2, M // 3, q) != 1 if L > 1 else True
print("2 primitive root ok")

def Sdist(k, M):
    g = np.array([2.0 ** (-r) for r in range(1, M + 1)]) / (1 - 2.0 ** (-M)) if M < 1000 else np.array([2.0 ** (-r) if r < 1100 else 0.0 for r in range(1, M + 1)])
    g = np.roll(g, 1)  # index r mod M: g[r % M] for r=1..M
    G = np.fft.fft(g)
    d = np.real(np.fft.ifft(G ** k))
    return d

worst_ratio2 = 0; worst_ratio4 = 0; worst_lclt = 0
for n in range(2, 11):
    for k in range(1, n):
        L = n - k; q = 3 ** L; M = 2 * 3 ** (L - 1)
        d = Sdist(k, M)
        inv2 = pow(2, -1, q)
        xi = units[n]
        # RHS of (*): sum_t d[t] |phi_L(xi 2^{-t} mod 3^L)|
        absphiL = np.abs(phi[L])
        rhs = np.zeros(len(xi)); rhs2 = np.zeros(len(xi))
        u = 1
        for t in range(M):
            idx = (xi * u) % q
            rhs += d[t] * absphiL[idx]
            rhs2 += d[t] * absphiL[idx] ** 2
            u = (u * inv2) % q
        lhs = np.abs(phi[n][xi])
        r2 = (lhs / rhs).max(); worst_ratio2 = max(worst_ratio2, r2)
        # C3
        assert Phi[n] <= Phi[L] + 1e-12, (n, L)
        # C4
        maxP = d.max()
        bound4 = maxP * I[L]
        r4 = (lhs ** 2).max() / bound4; worst_ratio4 = max(worst_ratio4, r4)
        lclt = 1.0 / M + 0.846 / np.sqrt(k)
        worst_lclt = max(worst_lclt, maxP / lclt)
        # check sum_t |phi_L(xi 2^-t)|^2 == I_L exactly
        u = 1; s = 0
        for t in range(M):
            s += absphiL[(int(xi[0]) * u) % q] ** 2; u = (u * inv2) % q
        assert abs(s - I[L]) < 1e-10
print("C2 worst lhs/rhs (<=1 means holds): %.6f" % worst_ratio2)
print("C4 worst |phi_n|^2/(maxP*I_L): %.6f" % worst_ratio4)
print("local-CLT bound worst maxP/(1/M+0.846/sqrt k): %.4f" % worst_lclt)
print("C3 ok for n<=10")
for n in range(2, NMAX + 1):
    for L in range(1, n):
        assert Phi[n] <= Phi[L] + 1e-12
print("C3 ok for n<=13")
# C5 shape: n^{1/4} Phi(n) and sqrt(n)*Phi^2
for n in range(2, NMAX + 1):
    print(n, "n^.25*Phi=%.4f" % (n ** 0.25 * Phi[n]), "Phi(n)/Phi(n-1)=%.4f" % (Phi[n] / Phi[n - 1]))
# asymptotic local CLT constant
for k in (10, 100, 1000):
    import math
    print(k, "sqrt(k)*max P(S=s) exact-ish", math.sqrt(k) * max(Sdist(k, 4096)), "1/sqrt(4pi)=", 1 / math.sqrt(4 * math.pi))
