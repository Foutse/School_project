from TPC04_methodes import *
import numpy as np
import matplotlib.pyplot as plt

#lecture de données
data=np.loadtxt('el_nino.mat')
temps=data[108:,0]
sst1=data[108:,1]
sst2=data[108:,2]
sst3=data[108:,3]
sst4=data[108:,4]

#histogrammes
plt.figure()
plt.hist(sst1)
plt.figure()
plt.hist(sst2)
plt.figure()
plt.hist(sst3)
plt.figure()
plt.hist(sst4)

#graphes des données
plt.figure()
plt.plot(temps,sst1,label='zone 1')
plt.plot(temps,sst2,label='zone 2')
plt.plot(temps,sst3,label='zone 3')
plt.plot(temps,sst4,label='zone 4')
plt.legend()
plt.xlabel('temps')
plt.ylabel('température')
plt.title('variation de la température au fil des années dans les 4 zones d études')

#moyennes 
moy1=np.mean(sst1)
moy2=np.mean(sst2)
moy3=np.mean(sst3)
moy4=np.mean(sst4)
moy=[moy1,moy2,moy3,moy4]

#écart type
ec1=np.std(sst1)
ec2=np.std(sst2)
ec3=np.std(sst3)
ec4=np.std(sst4)
ec=[ec1,ec2,ec3,ec4]

#boite a moustache
plt.figure()
plt.boxplot([sst1,sst2,sst3,sst4])

#diagramme de dispersion
plt.figure()
plt.plot(sst1,sst2,'b*')
plt.plot(sst1[132-108:144-108],sst2[132-108:144-108],'r*',label='année 1972')
plt.plot(sst1[264-108:276-108],sst2[264-108:276-108],'g*',label='année 1983')
plt.legend()
plt.figure()
plt.plot(sst1,sst3,'b*')
plt.plot(sst1[132-108:144-108],sst3[132-108:144-108],'r*',label='année 1972')
plt.plot(sst1[264-108:276-108],sst3[264-108:276-108],'g*',label='année 1983')
plt.legend()
plt.figure()
plt.plot(sst1,sst4,'b*')
plt.plot(sst1[132-108:144-108],sst4[132-108:144-108],'r*',label='année 1972')
plt.plot(sst1[264-108:276-108],sst4[264-108:276-108],'g*',label='année 1983')
plt.legend()
plt.figure()
plt.plot(sst3,sst2,'b*')
plt.plot(sst3[132-108:144-108],sst2[132-108:144-108],'r*',label='année 1972')
plt.plot(sst3[264-108:276-108],sst2[264-108:276-108],'g*',label='année 1983')
plt.legend()
plt.figure()
plt.plot(sst4,sst2,'b*')
plt.plot(sst4[132-108:144-108],sst2[132-108:144-108],'r*',label='année 1972')
plt.plot(sst4[264-108:276-108],sst2[264-108:276-108],'g*',label='année 1983')
plt.legend()
plt.figure()
plt.plot(sst3,sst4,'b*')
plt.plot(sst3[132-108:144-108],sst4[132-108:144-108],'r*',label='année 1972')
plt.plot(sst3[264-108:276-108],sst4[264-108:276-108],'g*',label='année 1983')
plt.legend()

#corrélation

#carte topologique

