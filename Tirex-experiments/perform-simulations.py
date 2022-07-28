#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:55:11 2022

@author: anass
"""

from scipy.stats import invweibull,expon,bernoulli
from math import floor,acos,degrees
import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import pareto
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
import time
from matplotlib.pyplot import figure

from TIREX_src import TIREX
np.random.seed(42)

matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({"font.family": "Avenir",   # specify font family here
    })



    
#----------------------------------------------------------------------------------

    


n_simu=10  #number of simulations

length=15  #  length of the grid for parameter k

theta=0.5        #parameter of te bernoulli 

   

alpha_low=10  # shape parameter of the exponential
alpha_high=10 # shape parameter of the pareto


p=30    #nbr of variables for example B

d=5     # SDR dimension for example B

def generate_example_A(n,q_bernoulli,alpha_low,alpha_high):
    X1=np.random.uniform(low=1,high=10,size=n)
    X2=np.random.uniform(low=1,high=10,size=n)
    
    
    B=bernoulli.rvs(q_bernoulli,size=n)
    
    T=np.random.exponential(scale=1/alpha_low,size=n)
    F=pareto.rvs(alpha_high,size=n)
    
    Z=np.zeros((n,2))
    
    Z[:,0]=F*X1
    Z[:,1]=T*X2
    y=Z[:,0]*B+(1-B)*Z[:,1] 
    X1=X1.reshape(-1,1)
    X2=X2.reshape(-1,1)
    y=y.reshape(-1,1)
    return np.concatenate((X1, X2,y), axis=1 )

def generate_example_B(n,q_bernoulli,alpha_low,alpha_high,p,d):
    
    X1=np.random.uniform(low=1,high=10,size=(n,d))
    X2=np.random.uniform(low=1,high=10,size=(n,p-d))

    
    B=bernoulli.rvs(q_bernoulli,size=n)
    # T=pareto.rvs(alpha_low,size=(n,p-d))
    T=np.random.exponential(scale=1/alpha_low,size=(n,p-d))
    F=pareto.rvs(alpha_high,size=(n,d))
   
    multinomial_low=np.random.choice([i for i in range(p-d)],size=(n,1))
    multinomial_high=np.random.choice([i for i in range(d)],size=(n,1))
    
    
    selected_attr_low=np.take_along_axis(X2,multinomial_low,axis=1)*np.take_along_axis(T,multinomial_low,axis=1)
    selected_attr_high=np.take_along_axis(X1,multinomial_high,axis=1)*np.take_along_axis(F,multinomial_high,axis=1)

    Z=np.zeros((n,2))
    Z[:,0]=selected_attr_high.ravel()
    Z[:,1]=selected_attr_low.ravel()
    y=Z[:,0]*B+(1-B)*Z[:,1] 
    X1=X1
    X2=X2
    y=y.reshape(-1,1)

    return np.concatenate((X1, X2,y), axis=1 )



def generate_example_C(n,q_bernoulli,alpha_low,alpha_high):

    X1=bernoulli.rvs(0.5,size=n)
    X2=bernoulli.rvs(0.5,size=n)
    
    
    B=bernoulli.rvs(q_bernoulli,size=n)
    

    T=np.random.exponential(scale=1/alpha_low,size=n)
    F=pareto.rvs(alpha_high,size=n)
    
    
    Z=np.zeros((n,2))
    
    Z[:,0]=F*X1
    Z[:,1]=T*X2
    y=Z[:,0]*B+(1-B)*Z[:,1] 
    X1=X1.reshape(-1,1)
    X2=X2.reshape(-1,1)
    y=y.reshape(-1,1)
    return np.concatenate((X1, X2,y), axis=1 )




    
def normalize(x):
    return x/ np.sqrt((x**2).sum(axis=0))    
    

def create_projector(e):
    return np.dot(e,e.T)
#----------------------------------------SIMulation when dim(Centrale_space)=1-----------------------------------------------



e_1=np.array([[1,0]]).reshape(-1,1)                         #true vector for models A and C
e_p=np.array([1 for j in range(d)]+[0 for j in range(p-d)]).reshape(-1,1)   #true vector for model B


e_p=np.diagflat(e_p)[:,:d]
    


#---creating projectors------


P_1=create_projector(e_1)
P_p=create_projector(e_p)




dict_toys={"Model A":(generate_example_A,P_1),"Model B":(generate_example_B,P_p),"Model C":(generate_example_C,P_1)}



for name,(generatrix,P) in dict_toys.items(): 
    print(f"Simulations for  {name}")
    
 

    list_distance_FO=np.zeros((n_simu,length))
    list_distance_SO=np.zeros((n_simu,length))
    
    list_projector_FO=[]
    list_projector_SO=[]
    
    
    #centrale space dimension
    if name!="Model B":
        dim=1 
        n=int(1e4)  
              
    else:
        dim=d
        n=int(1e5)  
   
    L_k_log=np.logspace(start=2,stop=np.log10(n),num=length,base=10)
    L_k=[int(k) for k in L_k_log] 
    
    for i in range(n_simu):
        print(f"STEP {i+1}/{n_simu}")
        if (name=="Model B") :
            sample=generatrix(n,theta,alpha_low,alpha_high,p,d)
            X,Y=sample[:,0:p],sample[:,p]
        else:
            sample=generatrix(n,theta,alpha_low,alpha_high)
            X,Y=sample[:,0:2],sample[:,2]
            
        n_f=X.shape[1]
        List_simu_project_FO=np.zeros((n_f,n_f,length))
        List_simu_project_SO=np.zeros((n_f,n_f,length))
        
      
        for j,k in enumerate(L_k):     
    
            SIR=TIREX(n_components=dim,k=k,method="FO",get_SDR_X=True)
            
            SIR.fit(X,Y)
            EC=SIR.CentralSpace_X
            
            eta=normalize(EC)
            P_eta=create_projector(eta)
            list_distance_FO[i,j]= np.linalg.norm(P_eta-P)**2
            List_simu_project_FO[:,:,j]=P_eta
            
            SIR=TIREX(n_components=dim,k=k,method="SO",get_SDR_X=True)
            SIR.fit(X,Y)
            EC=SIR.CentralSpace_X
            eta=normalize(EC)
            P_eta=create_projector(eta)
            list_distance_SO[i,j]= np.linalg.norm(P_eta-P)**2
            List_simu_project_SO[:,:,j] = P_eta
            
            

        list_projector_FO.append(List_simu_project_FO)
        list_projector_SO.append(List_simu_project_SO)
        arr_reshaped_FO = List_simu_project_FO.reshape(List_simu_project_FO.shape[0], -1)
        arr_reshaped_SO = List_simu_project_SO.reshape(List_simu_project_SO.shape[0], -1)

    E_p_FO=sum(list_projector_FO)/len(list_projector_FO)
    
    bias_FO=[np.linalg.norm(P-E_p_FO[:,:,j])**2 for j in range(E_p_FO.shape[2])]
    
    centered_projectors_norm=[np.linalg.norm(list_projector_FO[i]-E_p_FO,axis=(0,1))**2 for i in range(n_simu) ]
    variance_FO=sum(centered_projectors_norm)/len(centered_projectors_norm)
    
    MSE_FO=np.mean(list_distance_FO,axis=0)
    
    E_p_SO=sum(list_projector_SO)/len(list_projector_SO)
    
    bias_SO=[np.linalg.norm(P-E_p_SO[:,:,j])**2 for j in range(E_p_SO.shape[2])]
    
    centered_projectors_norm=[np.linalg.norm(list_projector_SO[i]-E_p_SO,axis=(0,1))**2 for i in range(n_simu) ]
    variance_SO=sum(centered_projectors_norm)/len(centered_projectors_norm)
    
    MSE_SO=np.mean(list_distance_SO,axis=0)
    
    bd_model=pd.DataFrame(list(zip(MSE_FO,bias_FO,variance_FO,MSE_SO,bias_SO,variance_SO)),columns=["MSE_FO","bias_FO","variance_FO","MSE_SO","bias_SO","variance_SO"])
    
    # bd_model.to_excel("RESULTS-small-n"+name+".xlsx") saving results if necessary
    L_sticks=[n//100,n//20,n//10,n//5,n]


    fig1, (ax1,ax2) = plt.subplots(1,2)
    fig1.suptitle(name,fontsize=50)
    fig1.set_size_inches(35, 10, forward=True)
    ax1.set_xscale('log')
    ax1.set_xlim(n//100,n)
    ax1.set_xlabel(r'$k$',fontsize=30)
    ax1.set_xticks(L_sticks)
    ax1.tick_params(labelsize=20)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.plot(L_k,MSE_FO,color="darkblue",label="MSE ",linewidth=5)
    ax1.plot(L_k,bias_FO,color="lawngreen",label="Bias ",linestyle="dashed",linewidth=5)
    ax1.plot(L_k,variance_FO,color="turquoise",label="Variance ",linewidth=5)
    ax1.set_title("TIRex1 ",fontsize=30)
    ax1.legend(loc="upper left", prop={'size': 40})
    
    ax2.set_xscale('log')
    ax2.set_xlim(n//100,n)
    
    ax2.set_xlabel(r'$k$',fontsize=30)
    ax2.set_xticks(L_sticks)
    ax2.tick_params(labelsize=20)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.plot(L_k,MSE_SO,color="darkblue",label="MSE ",linewidth=5)
    ax2.plot(L_k,bias_SO,color="lawngreen",label="Bias",linestyle="dashed",linewidth=5)
    ax2.plot(L_k,variance_SO,color="turquoise",label="Variance",linewidth=5)
    ax2.set_title("TIRex2",fontsize=30)
    ax2.legend(loc="upper left", prop={'size': 40})
    plt.minorticks_off()
    plt.savefig(name+"-Results")

    plt.show()
    plt.clf()   
    
    

    