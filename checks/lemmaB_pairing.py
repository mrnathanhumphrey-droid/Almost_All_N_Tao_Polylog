# Lemma B: exact pairing factorisation, the Cauchy-Schwarz steps, and inequality (4) for ten pairs (n, L).
import numpy as np, sys, time
AM=60; BM=44
def phi_all(n):
    # phi_n(xi) for all xi mod 3^n via phi_n(xi)=sum_a 2^-a e(xi 2^-a/3^n) phi_{n-1}(xi 2^-a mod 3^{n-1})
    ph=np.ones(1,dtype=complex)
    for m in range(1,n+1):
        q=3**m; xi=np.arange(q,dtype=np.int64); inv=pow(2,-1,q); u=xi.copy(); out=np.zeros(q,dtype=complex)
        for a in range(1,AM+1):
            u=(u*inv)%q
            out+=2.0**-a*np.exp(2j*np.pi*u/q)*ph[u%(q//3)]
        ph=out
    return ph
def run(n,L,phn,phL):
    q=3**n; M=2*3**(n-1); J=(n-L)//2; assert n-L==2*J and J>=1
    p2=np.empty(M,dtype=np.int64); c=1
    for u in range(M): p2[u]=c; c=c*2%q
    qL=3**L
    pb={b:(b-1)*2.0**-b for b in range(2,BM+1)}
    tailL=phL[p2%qL]            # phi_L(2^u)
    G=tailL.copy(); GA=np.abs(tailL); GQ=np.ones(M); GT=np.abs(tailL)**2
    for j in range(J,0,-1):
        c3=pow(3,2*j-2,q); base=c3*p2%q
        NG=np.zeros(M,dtype=complex); NA=np.zeros(M); NQ=np.zeros(M); NT=np.zeros(M); Ssum=np.zeros(M,dtype=complex)
        for b in range(2,BM+1):
            a=b-1; Ssum+=np.exp(2j*np.pi*(base*((pow(2,a,q)+3)%q)%q)/q)
            F=Ssum/(b-1)        # f(2^{u'} 3^{2j-2}, b) at state u'
            NG+=pb[b]*np.roll(F*G,b); NA+=pb[b]*np.roll(np.abs(F)*GA,b)
            NQ+=pb[b]*np.roll(np.abs(F)**2*GQ,b); NT+=pb[b]*np.roll(GT,b)
        G,GA,GQ,GT=NG,NA,NQ,NT
    tru=phn[p2]                 # phi_n(2^s)
    ident=np.abs(G-tru).max()
    v1=(np.abs(tru)>GA+1e-10).sum(); v2=(np.abs(tru)>np.sqrt(GT*GQ)+1e-10).sum()
    # point mass of D=a[1,n-L] mod M_L
    ML=2*3**(L-1); k=n-L; d=np.zeros(ML); d[0]=1
    for i in range(k):
        nd=np.zeros(ML)
        for a in range(1,AM+1): nd+=2.0**-a*np.roll(d,a)
        d=nd
    IL=(np.abs(phL[np.array([pow(2,t,qL) for t in range(ML)])])**2).sum()
    lclt=1/ML+0.846/np.sqrt(k)
    v3=(GT>d.max()*IL+1e-10).sum()
    ratio=np.abs(tru)/np.sqrt(GT*GQ)
    print("n=%d L=%d J=%d ident_err=%.1e viol(preCS,CS,T<=maxP*I)=%d,%d,%d Phi_n=%.4f maxsqrt(TQ)=%.4f tight max|phi|/sqrt(TQ)=%.3f median=%.3f maxP=%.4f bound=%.4f supQ=%.4f meanQ=%.6f 2^-J=%.6f F2rhs=%.4f"%(
        n,L,J,ident,v1,v2,v3,np.abs(tru).max(),np.sqrt(GT*GQ).max(),ratio.max(),np.median(ratio),d.max(),lclt,GQ.max(),GQ.mean(),2.0**-J,np.sqrt(lclt*IL*GQ.max())),flush=True)
cases=[(7,3),(9,5),(9,3),(11,3),(11,5),(12,4),(12,6),(13,5),(13,3),(13,7)]
t=time.time(); PH={}
for n in sorted({c[0] for c in cases}|{c[1] for c in cases}): PH[n]=phi_all(n)
print("phi done %.1fs"%(time.time()-t),flush=True)
for n,L in cases: run(n,L,PH[n],PH[L])
print("total %.1fs"%(time.time()-t))
