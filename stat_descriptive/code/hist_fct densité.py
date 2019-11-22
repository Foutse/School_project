import numpy as np
import math as ma
import matplotlib.pyplot as plt
import scipy.stats as sp

#génération du jeu de données
xn=np.random.randn(500)
x=3+xn*(ma.sqrt(5))

un=np.random.rand(500)
u=un*(5-3)+3

#verifier les caractéristiques des lois
moyN=np.mean(x)
varN=np.var(x)
minU=min(u)
maxU=max(u)

#les fonctions densité empirique
lin=np.linspace(0,10,100)
unif=sp.uniform.pdf(lin,3,ma.sqrt(5))

#tracé des hist
plt.subplot(2,2,1)
plt.hist(x)

plt.subplot(2,2,2)
plt.hist(x,normed=True)
plt.subplot(2,2,3)
plt.hist(u)
plt.subplot(2,2,4)
plt.hist(u,normed=True)
plt.plot(lin,unif)


