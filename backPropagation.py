# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:34:10 2020

@author: bo.pei
"""

import numpy as np
import pandas as pd
from random import random
from random import seed

    
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

inputs=np.array([1,1])

# Construct the structure of the network:
'''
    Weights: row: Number of neurons in the next layer
             column: Number of neurons in the previous layer
'''

def build_layer(n_prev,n_next):
    layer=[{'weights':[np.random.random() for i in range(n_prev)],'b':np.random.random()}
           for i in range(n_next)]
    return layer

def initialize_network(layers=3,structure=[2,3,2]):
    if len(structure)!=layers:
        print("Network structure error...")
        return
    else:
        network=list()
        for i in range(layers-1):
            layer=build_layer(structure[i],structure[i+1])
            network.append(layer)
    return network

def activate(weights,inputs,b):
    activation=np.dot(weights,inputs)+b
    return activation

def transfer(activation):
    return 1/(1+np.exp(-activation))

def testIToH(inputs,layer):
    Layerdf=pd.DataFrame(layer)
    w=np.array([np.array(i) for i in Layerdf['weights'].to_numpy()])
    b=np.array([np.array(i) for i in Layerdf['b'].to_numpy()])
    netValue=activate(w,inputs,b)
    return netValue

# Forward Propagation 
    '''
        LayerParams:the parameters that connect the two layers
    '''
network=initialize_network()
def forward_propagation(inputs=inputs,network=network):
    network=network
    
    #forward process
    LayerParams=[]
    for layer in range(len(network)):
        df=pd.DataFrame(network[layer])
        weights=np.array([np.array(w) for w in df['weights'].to_numpy()])
        bias=np.array([np.array(b) for b in df['b'].to_numpy()])
        df['netValue']=activate(weights,inputs,bias)
        df['outputValue']=transfer(df['netValue'])
        inputs=df['outputValue'].values
        LayerParams.append(df)
    return LayerParams

# Calculate the delta for each layer, basically it is the derivation of netOutput of each layer
def delta(LayerParams,expected):    
    
    # Calculate the delta for the last layer
    df=LayerParams[-1]
    df['delta']=(expected-df.outputValue)*df.outputValue*(1-df.outputValue)
    LayerParams[-1]=df
    
    # Calculate the delta for the rest of the layers
    for i in reversed(range(len(LayerParams)-1)):
        df=LayerParams[i] # current layer        
        nextLayer=LayerParams[i+1] #next layer        
        arrayW=[]
        for w in nextLayer['weights'].to_numpy():
            arrayW.append(np.array(w))
            
        arrayW=np.array(arrayW)
        
        # convert the values to the column values
        nextDeltaVal=nextLayer['delta'].values[:,np.newaxis]
        
        #calculate the derivation with respect to output of the previous layer
        tempDeltaCur=np.sum(arrayW*nextDeltaVal,axis=0) 
        
        #calculate the derivation with repect to the net output to the previous layer
        df['delta']=tempDeltaCur*df['outputValue'].values*(1-df['outputValue'].values)  
        LayerParams[i]=df
    return LayerParams

def update_weights(inputs,LayerParams,l_rate):
    currLayer=LayerParams[0]
    updated=currLayer['delta'].values[:,np.newaxis]*l_rate*inputs
#    currLayer['weights']+=updated
    currW=np.array([np.array(w) for w in currLayer['weights']])
    currW+=updated
    currLayer['weights']=currW.tolist()
    
    updatedB=currLayer['b'].values+currLayer['delta'].values*l_rate
    currLayer['b']=updatedB.tolist()
    
    LayerParams[0]=currLayer
    
    for i in range(1,len(LayerParams)):
        inputs=LayerParams[i-1]['outputValue'].values
        currLayer=LayerParams[i]
        updated=currLayer['delta'].values[:,np.newaxis]*l_rate*inputs
#        currLayer['weights']+=updated
        currW=np.array([np.array(w) for w in currLayer['weights']])
        currW+=updated
        currLayer['weights']=currW.tolist()
        updatedB=currLayer['b'].values+currLayer['delta'].values*l_rate
        currLayer['b']=updatedB.tolist()
        LayerParams[i]=currLayer
    
    # modify the weights in the network
    newNetwork=[]
    for i,layer in enumerate(network):
        newLayer=[]
        for idx,neuron in enumerate(layer):
            neuron['weights']=LayerParams[i]['weights'].iloc[idx]
            neuron['b']=LayerParams[i]['b'].iloc[idx]
            newLayer.append(neuron)
        newNetwork.append(newLayer)
        
        
    return LayerParams,newNetwork

# train the network
seed(1);
targets=[data[-1] for data in dataset]
data=[d[:-1] for d in dataset]


sum_error=[]
n_epoch=1500

network=initialize_network(layers=3,structure=[2,3,2])

for epoch in range(n_epoch):
    errors=0
    for i in range(len(data)):
        expected=np.array([0,0])
        inputs=np.array(data[i])
        expected[targets[i]]=1
        LayerParams=forward_propagation(inputs,network)
        
        outputs=LayerParams[-1]['outputValue'].values
        errors+=sum((expected-outputs)**2)
        
        LayerParams=delta(LayerParams,expected)
        LayerParams,network=update_weights(inputs,LayerParams,0.001)
        
    
    sum_error.append(errors)


import matplotlib.pyplot as plt
plt.plot(range(1500),sum_error,'r--')
        
        



    
