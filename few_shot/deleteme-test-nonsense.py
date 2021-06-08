# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:47:53 2021

@author: Support
"""
import numpy as np
import random




npzfile = np.load("data/SETA.npz", allow_pickle=True)
data = npzfile["data"]
labels = npzfile["labels"]
npzfile.close()
distinctLabelCount= len(set(labels))

print("data", data.shape)
print("labels", labels.shape)
print("distinctLabelCount", distinctLabelCount)

random.seed(123)
testClasses = random.sample(range(0,distinctLabelCount),7)
testClasses = [np.float64(i) for i in testClasses]
mask = np.isin(labels,testClasses)
print("Mask", mask)
print(type(mask))
worstClasses = [18.0,16.0,14.0,11.0,10.0,9.0,7.0]
print(worstClasses)
print("teasrsa", type(worstClasses))
