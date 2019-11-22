import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
import math as ma
import scipy.stats as ss

l=[]
lg=[]

p=200
n=500
for _ in range(p):
    u=npr.rand(n)
    u1=2*u+3
    moy=np.mean(u1)
    l.append(moy)
lg.append(l)

m1=np.mean(lg)
v1=np.std(lg)



plt.title('histogramme normalisé des moyennes ')
plt.hist(lg,normed=1) 


x1=np.linspace(0,10,100)

plt.title('fonction de densité empirique')
plt.plot(x1,ss.norm.pdf(x1,4,0.015))

