# Lemma C: C_L <= 2(1 + L/3)(sqrt(2L+1) + 4.45)^2 for L <= 12; brute-force check of Steps 1-3 for L <= 6;
# the tail constant sqrt(2/3)/(1 - sqrt(2/3)) = 4.449...
import numpy as np, itertools, math
from fractions import Fraction
# Exact-in-structure distribution of X_L via Y_L = 2^{-a1}(1+3*Y'_{L-1}); 2 is a primitive root mod 3^L.
def dist(L):
    P = np.array([1.0])            # L=0: Y=0 mod 1
    for l in range(1, L+1):
        m = 3**l; p = 2*3**(l-1)
        lg = -np.ones(m, dtype=np.int64); v = 1
        for e in range(p): lg[v] = e; v = (v*2) % m
        z = (1 + 3*np.arange(3**(l-1))) % m
        ez = np.zeros(p); np.add.at(ez, lg[z], P)
        w = 2.0**(-np.arange(p, dtype=float)); w[0] = 2.0**(-p); w /= w.sum()   # P(a = r mod p) for a ~ Geom(2) on {1,2,...}
        # x = 2^{e - r}: distribution over exponent e-r
        wr = np.zeros(p); wr[0] = w[0]; wr[1:] = w[1:][::-1]  # wr[s] = w[-s mod p]
        out = np.fft.irfft(np.fft.rfft(ez)*np.fft.rfft(wr), n=p)
        out = np.maximum(out, 0)
        pw = np.zeros(p, dtype=np.int64); v=1
        for e in range(p): pw[e]=v; v=(v*2)%m
        Pn = np.zeros(m); Pn[pw] = out
        P = Pn
    return P
def bound(L): return 2*(1+L/3)*(math.sqrt(2*L+1)+4.45)**2
for L in range(1, 13):
    P = dist(L); C = 3**L*np.sum(P*P)
    print(L, "sum", round(P.sum(),12), "C_L", round(C,6), "bound", round(bound(L),2), "ok", C <= bound(L))
# brute checks for small L: enumerate tuples with D=d, verify injectivity of N, odd, and per-d inequality 3^L||f_d||^2 <= 2 beta_d
def Nof(t):
    L=len(t); return sum(3**(j-1)*2**sum(t[j:]) for j in range(1,L+1))
bad=0
for L in range(1,7):
    m=3**L
    for d in range(L, L+12):
        tuples=[t for t in itertools.product(range(1,d-L+2),repeat=L) if sum(t)==d]
        Ns=[Nof(t) for t in tuples]
        assert len(set(Ns))==len(Ns) and all(n%2==1 for n in Ns)
        # decode check
        for t,N in zip(tuples,Ns):
            r=[];NN=N
            for l in range(L,1,-1):
                q=NN-3**(l-1); a=(q & -q).bit_length()-1; r.append(a); NN=q>>a
            assert NN==1 and tuple([d-sum(r)]+r[::-1])==t
        inv=pow(2,-d,m); cnt={}; Ssum={}
        for N in Ns:
            x=N*inv%m; cnt[x]=cnt.get(x,0)+1; Ssum[x]=Ssum.get(x,0)+N
        lhs=Fraction(m)*sum(Fraction(c,2**d)**2 for c in cnt.values())
        Qd=Fraction(m*len(tuples),4**d); EY=Fraction(sum(Ns),4**d)
        if not lhs<=2*(Qd+EY): bad+=1; print("FAIL",L,d)
        # step2 lower bound
        for x in cnt:
            n=cnt[x]; assert Ssum[x]>=m*n*(n-1)
print("brute step1-3 failures:",bad)
# tail constant
s=math.sqrt(2/3); print("tail const", s/(1-s))
