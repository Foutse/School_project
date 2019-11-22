import numpy as np
import matplotlib.pyplot as mtp

px=[]
x=np.random.randint(1,7,10)

h,b=np.histogram(x,bins=np.arange(0.5,6.51,1))
px.append(h)

mtp.figure()
mtp.title('10 lancés')
mtp.pie(h,labels=['classe 1','classe 2','classe 3','classe 4','classe 5','classe 6'],colors=['skyblue','pink','orange','yellow','lightgreen','red'],autopct='%1.1f%%')

x=np.random.randint(1,7,20)
h,b=np.histogram(x,bins=np.arange(0.5,6.51,1))
px.append(h)

mtp.figure()

mtp.title('20 lancés')
mtp.pie(h,labels=['classe 1','classe 2','classe 3','classe 4','classe 5','classe 6'],colors=['skyblue','pink','orange','yellow','lightgreen','red'],autopct='%1.1f%%')

x=np.random.randint(1,7,50)

h,b=np.histogram(x,bins=np.arange(0.5,6.51,1))
px.append(h)

mtp.figure()

mtp.title('50 lancés')
mtp.pie(h,labels=['classe 1','classe 2','classe 3','classe 4','classe 5','classe 6'],colors=['skyblue','pink','orange','yellow','lightgreen','red'],autopct='%1.1f%%')


x=np.random.randint(1,7,100)

h,b=np.histogram(x,bins=np.arange(0.5,6.51,1))
px.append(h)

mtp.figure()
mtp.title('100 lancés')
mtp.pie(h,labels=['classe 1','classe 2','classe 3','classe 4','classe 5','classe 6'],colors=['skyblue','pink','orange','yellow','lightgreen','red'],autopct='%1.1f%%')

x=np.random.randint(1,7,500)

h,b=np.histogram(x,bins=np.arange(0.5,6.51,1))
px.append(h)

mtp.figure()

mtp.title('500 lancés')
mtp.pie(h,labels=['classe 1','classe 2','classe 3','classe 4','classe 5','classe 6'],colors=['skyblue','pink','orange','yellow','lightgreen','red'],autopct='%1.1f%%')

x=np.random.randint(1,7,2000)
h,b=np.histogram(x,bins=np.arange(0.5,6.51,1))
moyenne=np.mean(x)

mtp.figure()
mtp.title('2000 lancés')
mtp.pie(h,labels=['classe 1','classe 2','classe 3','classe 4','classe 5','classe 6'],colors=['skyblue','pink','orange','yellow','lightgreen','red'],autopct='%1.1f%%')

print(moyenne)

freq=[]
for i in range(0,6):
    fr=h1[i]/np.sum(h1)
    freq.append(fr)
    
max(freq)
min(freq)
np.std(freq)