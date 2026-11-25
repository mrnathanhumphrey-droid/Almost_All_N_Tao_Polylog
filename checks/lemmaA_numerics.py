# Lemma A numerics (Section 9.2): Monte Carlo K*P(S_inf > K) (critical 1/K tail), the moment E 2^{-qa} formula,
# and sum_k sqrt(P(u = k)) / n^{1/4} ~ 1.78 for n up to 3200 (Remark 5.1).
import numpy as np, math, cmath
rng=np.random.default_rng(1)
# 1) perpetuity tail S=sum_j 3^{j-1} 2^{-a[1,j]}
N=400000; J=200
S=np.zeros(N); P=np.ones(N)/3.0
for j in range(J):
    a=rng.geometric(0.5,N); P*=3.0*2.0**(-a); S+=P
for K in [10,30,100,300,1000]:
    print("K",K,"P(S>K)*K",(S>K).mean()*K)
for s in [0.5,0.9,0.99]: print("s",s,"E M^s",3**s/(2**(s+1)-1))
# 2) monotonicity of Phi via exact law (Lemma 1.12 style recursion X_{L+1}=2^{-a}(1+3X_L))
def law(L):
    p=np.array([1.0])
    for n in range(L):
        q=np.zeros(3**(n+1)); mod=3**(n+1)
        xs=(1+3*np.arange(3**n))%mod
        inv2=pow(2,-1,mod)
        for a in range(1,2*3**(n+1)+1):
            xs=(xs*inv2)%mod
            np.add.at(q,xs,p*2.0**(-a))
        p=q/q.sum()
    return p
prev=None
for L in range(1,10):
    p=law(L); f=np.fft.fft(p); idx=[x for x in range(3**L) if x%3]
    phi=max(abs(f[i]) for i in idx); print("L",L,"Phi",round(phi,4),"L^.25*Phi",round(phi*L**.25,4))
# 3) stopping time k: first k with a[1,k+1]>=T, T=ceil(n log2 3); sum_k sqrt P(k)
for n in [200,800,3200]:
    T=math.ceil(n*math.log2(3))
    dist=np.zeros(T); dist[0]=1.0  # law of a[1,k] restricted to <T
    tot=[]
    for k in range(n):
        # P(k_*=k)=sum_{t<T} P(a[1,k]=t, path stayed<T) * P(a_{k+1}>=T-t)
        t=np.arange(T); pk=(dist*2.0**(-(T-t-1)).clip(max=1)).sum() if True else 0
        pk=(dist*np.minimum(1,2.0**(-(T-t-1)))).sum(); tot.append(pk)
        nd=np.zeros(T)
        for a in range(1,60):
            nd[a:]+=dist[:T-a]*2.0**(-a)
        dist=nd
    tot=np.array(tot); print("n",n,"sum P",round(tot.sum(),4),"sum sqrtP / n^.25",round(np.sqrt(tot).sum()/n**.25,3))
