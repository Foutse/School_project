# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 22:27:34 2018
"""

def unique(tabforfor):

    indice=[]
    for i in range (132):
            indice.append(i)
        
        #récupération des indices    
    formefor=[]
    for i in range (132):
        if i==indice[0]:
            temp=[]
            for j in indice:
                if list(tabforfor[i,:])==list(tabforfor[j,:]):
                    temp.append(j)
            formefor.append(temp)  
            a=len(temp)
            for k in range(a-1,-1,-1):
                indice.pop(indice.index(temp[k]))
            if len(indice)<1:
                break

    return formefor


