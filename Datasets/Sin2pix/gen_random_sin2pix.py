# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:12:57 2021

@author: mmanivan

Generate Synthetic data set used in 

    Pattern Recognition and Machine Learning
    
    Christopher M. Bishop
    
y = Sin (2*pi*x) + Normal(0,sd)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gen_random_sin2pix(num_values,std_dev):
    """
    Generate a random data set of outputs=sin(2*pi*inputs)+noise
    
    where 
    
        inputs - an array of size num_values 
            uniformly randomly generated from the interval 0,1
        
        noise - an array of size num_values
            randomly generated from the normal distribution 
            centred at 0 
            spread with a standard deviation of std_dev
                
        targets - sin(2*pi*inputs)
        
    returns
    
        input,outputs,targets
    
    """
    inputs=np.random.rand(num_values)
    
    noise=np.random.randn(num_values)*std_dev
    
    targets=np.sin(2.0*np.pi*inputs)
    
    outputs=targets+noise
    
    return [inputs,outputs,targets]


if __name__ == "__main__":
    
    
    num_values=1000
    std_dev=0.3
    num_points_to_plot=15
    inputs,outputs,targets=gen_random_sin2pix(num_values,std_dev)

    df=pd.DataFrame(np.array([inputs,outputs,targets]).T,columns=['Input','Output','Target'])
    df.to_csv('Synthetic_data_set.csv',index=False)
    
    df_unsort=df.copy()
    df.sort_values(by=['Input'],inplace=True)
    
    
    # Show the scatter of some data points about the target curve
    plt.figure(1,figsize=[10,10])
    plt.scatter(df_unsort.loc[:num_points_to_plot,'Input'],df_unsort.loc[:num_points_to_plot,'Output'],s=100,lw=2,facecolors=None,edgecolors='blue',alpha=0.5)
    plt.plot(df['Input'],df['Target'],linewidth=2)
    
    # Show the Error Displacement
    plt.figure(2,figsize=(10,10))
    plt.plot(df['Input'],df['Target'],linewidth=2)
    plt.scatter(df_unsort.loc[:num_points_to_plot,'Input'],df_unsort.loc[:num_points_to_plot,'Output'],s=100,lw=2,facecolors=None,edgecolors='blue',alpha=0.5)
    for iind in range(num_points_to_plot+1):
        plt.plot([df_unsort.loc[iind,'Input'],df_unsort.loc[iind,'Input']],[df_unsort.loc[iind,'Output'],df_unsort.loc[iind,'Target']],color='red')