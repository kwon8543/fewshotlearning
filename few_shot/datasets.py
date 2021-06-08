from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os
import sys

from config import DATA_PATH


class AWF(Dataset):
    def __init__(self, subset):
        """Dataset class representing AWF dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['data']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        #print("item",item)
        instance = self.datasetid_to_filepath[item]
        # Reindex to channels first format as supported by pytorch
        #instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        #print("\n", instance, "\n")
        
        instance = (instance - instance.min()) / (sys.float_info.epsilon + instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())


    @staticmethod
    def index_subset(subset, numTestClasses=100, randomSeed=123):
        """Index a subset by looping through all of its files and recording relevant information.
        
        # Arguments
            subset: Must be one of (background, evaluation)
            scaler: Standard, MinMax or "" (no scaling)

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            AWF dataset
        """
        print("datasets.py is ingesting the AWF dataset from data/AWF.npz")
        npzfile = np.load("data/AWF.npz", allow_pickle=True)
        data = npzfile["data"]
        
        labels = npzfile["labels"]
        
        print("The dataset shape is", data.shape)
        print("The labels shape is", labels.shape)
        npzfile.close()
        distinctLabelCount= len(set(labels))
        print("The total labels are", distinctLabelCount)
        
        
        distinctLabels= set(labels)
        #random.seed(randomSeed)
        selectedClasses = random.choices(tuple(distinctLabels), k=numTestClasses)
        
        
        dfLabels = pd.DataFrame(labels)
        mask = dfLabels[0].isin(selectedClasses)
        
        X_test = data[np.where(mask)]
        X_train = data[np.where(~mask)]
        y_test = labels[np.where(mask)]
        y_train = labels[np.where(~mask)]
            
        if subset=="background":
            print("\nCurrent working directory is ", os.getcwd())
            print("\n\n________________________________")
            print("Assembling the training dataset")
            print("________________________________")
            X = X_train
            y = y_train
            print("\n\nAWF X_train data shape", X.shape)
            print("AWF y_train data shape", y.shape)
            
        elif subset=="evaluation":
            print("\n\n________________________________")
            print("Assembling the test dataset")
            print("________________________________")            
            print("\nThe number of test classes is:", len(selectedClasses))
            print("\n\nThe test classes are:\n\n", selectedClasses)
            
            X = X_test
            y = y_test
            print("\n\nAWF X_test data shape", X.shape)
            print("AWF y_test data shape", y.shape)
        images = []
        
        #print("y shape", y.shape)
        
        
        for i in range(y.shape[0]):
            images.append({
                'subset': subset,
                'class_name': y[i],
                'data': X[i].reshape(1,X[1].shape[0])   # This is the shape needed for Matching Networks
                #'data': X[i] #.reshape(X[1].shape[0])  # This is the shape needed for MAML
                })
        
        return images
    
    
    
   
class DF(Dataset):
    def __init__(self, subset):
        """Dataset class representing SETA dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['data']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        #print("item",item)
        instance = self.datasetid_to_filepath[item]
        # Reindex to channels first format as supported by pytorch
        #instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        #print("\n", instance, "\n")
        instance = (instance - instance.min()) / (sys.float_info.epsilon + instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())


    @staticmethod
    def index_subset(subset, numTestClasses=33, randomSeed=123):
        """Index a subset by looping through all of its files and recording relevant information.
        
        # Arguments
            subset: Must be one of (background, evaluation)
            scaler: Standard, MinMax or "" (no scaling)

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            DF dataset
        """
        print("datasets.py is ingesting the DF dataset from data/DF.npz")
        npzfile = np.load("data/DF.npz", allow_pickle=True)
        data = npzfile["data"]
        labels = npzfile["labels"]
        npzfile.close()
        distinctLabelCount= len(set(labels))
        
        print("\ndata", data.shape)
        print("labels", labels.shape)
        print("Total distinct class labels in the DF.npz dataset: ", distinctLabelCount)
        
        #random.seed(randomSeed)
        testClasses = random.sample(range(0,distinctLabelCount),numTestClasses)
        testClasses = [np.float64(i) for i in testClasses]
        
        
        mask = np.isin(labels,testClasses)

        X_test = data[np.where(mask)]
        X_train = data[np.where(~mask)]
        y_test = labels[np.where(mask)]
        y_train = labels[np.where(~mask)]
            
        if subset=="background":
            print("Current working directory is ", os.getcwd())
            X = X_train
            y = y_train
            print("DF background data shape", X.shape)
        elif subset=="evaluation":
            X = X_test
            y = y_test
            print("DF evaluation data shape", X.shape)
        images = []
        
        #print("y shape", y.shape)
        
        
        for i in range(y.shape[0]):
            images.append({
                'subset': subset,
                'class_name': y[i],
                'data': X[i].reshape(1,X[1].shape[0])   # This is the shape needed for Matching Networks
                #'data': X[i] #.reshape(X[1].shape[0])  # This is the shape needed for MAML
                })
        
        return images
      
    
class DC(Dataset):
    def __init__(self, subset):
        """Dataset class representing DC dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['data']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        #print("item",item)
        instance = self.datasetid_to_filepath[item]
        # Reindex to channels first format as supported by pytorch
        #instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        #print("\n", instance, "\n")
        instance = (instance - instance.min()) / (sys.float_info.epsilon + instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())


    @staticmethod
    def index_subset(subset, numTestClasses=3, randomSeed=123):
        """Index a subset by looping through all of its files and recording relevant information.
        
        # Arguments
            subset: Must be one of (background, evaluation)
            scaler: Standard, MinMax or "" (no scaling)

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            DC dataset
        """
        print("datasets.py is ingesting the DF dataset from data/DC.npz")
        npzfile = np.load("data/DC.npz", allow_pickle=True)
        data = npzfile["data"]
        labels = npzfile["labels"]
        npzfile.close()
        distinctLabelCount= len(set(labels))
        
        #print("data", data.shape)
        #print("labels", labels.shape)
        #print("distinctLabelCount", distinctLabelCount)
        
        #random.seed(randomSeed)
        testClasses = random.sample(range(0,distinctLabelCount),numTestClasses)
        testClasses = [np.float64(i) for i in testClasses]
        
        
        mask = np.isin(labels,testClasses)

        X_test = data[np.where(mask)]
        X_train = data[np.where(~mask)]
        y_test = labels[np.where(mask)]
        y_train = labels[np.where(~mask)]
            
        if subset=="background":
            print("Current working directory is ", os.getcwd())
            X = X_train
            y = y_train
            print("DC background data shape", X.shape)
        elif subset=="evaluation":
            X = X_test
            y = y_test
            print("DC evaluation data shape", X.shape)
        images = []
        
        #print("y shape", y.shape)
        
        
        for i in range(y.shape[0]):
            images.append({
                'subset': subset,
                'class_name': y[i],
                'data': X[i].reshape(1,X[1].shape[0])   # This is the shape needed for Matching Networks
                #'data': X[i] #.reshape(X[1].shape[0])  # This is the shape needed for MAML
                })
        
        return images
    
       
    
    
class SETA(Dataset):
    def __init__(self, subset):
        """Dataset class representing SETA dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))
        #print(self.df.shape())
        
        
        
        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['data']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        #print("item",item)
        instance = self.datasetid_to_filepath[item]
        # Reindex to channels first format as supported by pytorch
        #instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        #print("\n", instance, "\n")
        instance = (instance - instance.min()) / (sys.float_info.epsilon + instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())


    @staticmethod
    def index_subset(subset, numTestClasses=7, randomSeed=123 ):
        """Index a subset by looping through all of its files and recording relevant information.
        
        # Arguments
            subset: Must be one of (background, evaluation)
            scaler: Standard, MinMax or "" (no scaling)

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            SETA dataset
        """
        print("datasets.py is ingesting the DF dataset from data/SETA.npz")
        npzfile = np.load("data/SETA.npz", allow_pickle=True)
        data = npzfile["data"]
        labels = npzfile["labels"]
        npzfile.close()
        distinctLabelCount= len(set(labels))
        

        #print("distinctLabelCount", distinctLabelCount)
        
        #random.seed(randomSeed)
        testClasses = random.sample(range(0,distinctLabelCount),numTestClasses)

        # Test the overrides 
       
        # WORST 
        #testClasses = np.array([18.0,16.0,14.0,11.0,10.0,9.0,7.0]).tolist()
        # BEST
        #testClasses = np.array([5.0,6.0,8.0,12.0,13.0,14.0,19.0]).tolist()
        
        testClasses = [np.float64(i) for i in testClasses]

        mask = np.isin(labels,testClasses)
        
        
        X_test = data[np.where(mask)]
        X_train = data[np.where(~mask)]
        y_test = labels[np.where(mask)]
        y_train = labels[np.where(~mask)]
            
        if subset=="background":
                    
            print("SETA data", data.shape)
            print("labels", labels.shape)
            print(testClasses)
            
            print("Current working directory is ", os.getcwd())
            X = X_train
            y = y_train
            print("SETA background data shape", X.shape)
        elif subset=="evaluation":
            X = X_test
            y = y_test
            print("SETA evaluation data shape", X.shape)
        images = []
        
        #print("y shape", y.shape)
        
        
        for i in range(y.shape[0]):
            images.append({
                'subset': subset,
                'class_name': y[i],
                'data': X[i].reshape(1,X[1].shape[0])   # This is the shape needed for Matching Networks
                #'data': X[i] #.reshape(X[1].shape[0])  # This is the shape needed for MAML
                })
        
        return images
        
class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (sys.float_info.epsilon + instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        
        
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,                    
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)
