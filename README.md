This repository is a modified version of Oscar Knagg's few-shot learning code for use with 1D network traffic fingerprinting analysis.

Oscar's repo is here: https://github.com/oscarknagg/few-shot

Although there are three models implemented in the code originally (MAML, Matching networks and Prototypical networks), currently only the matching networks code has been updated to work with the SETA, DF, DC and AWF datasets.

The original code was developed for few-shot image classification on Omniglot and MiniImageNet.
That code is still in this repo, however all the 2D convolutions were converted to 1D convolutions, so if you want to run these algorithms on Omniglot and MiniImageNet, it's probably just as easy to grab Oscar's 

# Quick start
Once you have Pytorch installed etc (you need an NVIDIA GPU, by the way), there is a Windows batch file called testSETA_matchingnetworks.bat that you can use to test the model is working.

# Hyper-parameter settings

These settings are configured within the file experiments/matching_nets.py:
1) The evaluation episodes 
2) The episodes per epoch
3) The LSTM input size (set to 5000 for DF, 500 for DC , 5000 for AWF and 500 for SETA)
4) The number of epochs
5) The number of input channels (e.g. 1 for our time series packet trace data, 3 for colour images)

These settings are configured within the file few_shot/datasets.py:
1) The number of test classes
2) The location of the four datasets (e.g. data/AWF.npz)

These settings are configured within the file few_shot/models.py:
1) The attention models g and f (currently a bidirectional LSTM and LSTM with attention respectively)


# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

