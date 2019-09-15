
# coding: utf-8

# In[ ]:


import numpy as np
import h5py
import sys, inspect
import os
import math


# In[ ]:


#Function:generate data from CGAN.h5 with especific batch 
#dataset:String => especific dataset "train", "test", "dev" from CGAN.h5 dataset
#batchSize:Int => number of examples generated from dataset
#pathH5pyFile:String => path of CGAN.h5 file
def generateHdf5Data(dataset, batchSize, pathH5pyFile=None):
    if pathH5pyFile == None:
        currentPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        pathH5pyFile = os.path.join(currentPath, "preprocess data/CGAN.h5")
    file = h5py.File(pathH5pyFile, 'r+')
    DataInput = file["{0}/input/imgs".format(dataset)]
    DataOutput = file["{0}/output/imgs".format(dataset)]
    
    lenDataInput = DataInput.shape[0]
    lenDataOutput = DataOutput.shape[0]
    assert lenDataInput ==lenDataOutput
    batchs = list(np.arange(0, int(math.ceil(lenDataInput/batchSize))))
    for batch in batchs:
        init = batch*batchSize
        end = min([(init+batchSize), lenDataInput])
        yield [DataInput[init:end,...], DataOutput[init:end,...]]
    file.close()
#currentPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
#pathH5pyFile = os.path.join(currentPath, "preprocess data/CGAN.h5")
#dataset = "dev"
#batchSize = 100

