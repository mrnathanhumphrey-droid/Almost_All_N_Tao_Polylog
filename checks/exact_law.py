# Exact law of X_n = Syrac(Z/3^n Z) for n <= 13 via the recursion of [Tao, Lemma 1.12].
# Prints Phi(n) (Table 3), C_n = 3^n sum P^2 (Table 4) and I_n = sum over units |phi_n|^2 (Section 9.3).
import numpy as np
# P_n on Z/3^n of X_n = sum_{j=1}^n 3^{j-1} 2^{-a[1,j]}, a iid Geom(2) on {1,2,..}
# recursion X_n = 2^{-a1}(1+3X'_{n-1})
NMAX=13
P={1:None}
M=3; inv2=pow(2,-1,M)
# n=1: X_1=2^{-a} mod 3 : 2^{-a} = 2 if a odd, 1 if a even ; P(a odd)=2/3
P1=np.zeros(3); P1[2]=2/3; P1[1]=1/3
P[1]=P1
for n in range(2,NMAX+1):
    M=3**n; prev=P[n-1]; Mp=3**(n-1)
    # distribution of Z=1+3X' mod 3^n
    Z=np.zeros(M); idx=(1+3*np.arange(Mp))%M; Z[idx]=prev
    # X = 2^{-a} Z ; sum over a: X = h^a Z, h=2^{-1}; orbit length N=2*3^{n-1}
    N=2*Mp; h=pow(2,-1,M)
    # Q(Y)=sum_a 2^{-a} Z(h^{-a} Y)=sum_a 2^{-a} Z(2^a Y); use recursion over a cyclic: Q = (1/(1-2^-N)) sum_{a=1}^N 2^{-a} Z(2^a Y)
    # do doubling: T_k(Y)=sum_{a=1}^{2^k}... simpler iterate geometric recursion: Q(Y)=1/2 Z(2Y)+1/2 Q(2Y). fixed point iteration via powers
    Y=np.arange(M)
    perm=(2*Y)%M
    # compute S=sum_{a=1}^{K} 2^{-a} Z∘(2^a) with K large via binary doubling
    A=0.5*Z[perm]; mult=perm.copy(); w=0.5  # A = sum_{a=1}^{1} ..., mult = map Y->2^1 Y, w=2^-1
    K=1
    while K<60:
        # A_{2K} = A_K + w_K * A_K∘mult_K
        A=A+w*A[mult]; mult=mult[mult]; w=w*w; K*=2
    P[n]=A
    s=A.sum()
    F=np.fft.fft(A)  # F[xi]=sum P(Y) e(-xi Y/M)
    units=np.array([x%3!=0 for x in range(M)])
    ab=np.abs(F)
    Phi=ab[units].max()
    coll=M*(A**2).sum()
    Iu=(ab[units]**2).sum()
    print(n, round(s,12), "Phi=%.5f"%Phi, "3^n sumP^2=%.4f"%coll, "units-sum|phi|^2=%.4f"%Iu, "mean_unit|phi|^2*3^n=%.4f"%(Iu/units.sum()*M), flush=True)
