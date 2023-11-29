import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import ToTensor
import pandas as pd
import os
import numpy as np


'''
############ EXAMPLE USAGE:

csv_file = '../data/movie_data.csv'
img_dir = '../data/MoviePosters'
pod = PosterDataset(csv_file, img_dir, genres = ['Action', 'Animation'])

############ CAN USE THESE FUNCTIONS, BY PYTORCH DATALOADER WILL USE THEM FOR YOU

pod.__getitem__(0)
len(pod)

############ CREATE A TRAINING AND TESTING DATALOADER TO ITERATE OVER LATER

trainSize = int(len(pd) * 0.8)
testSize = len(pd) - trainSize

trainData, testData = torch.utils.data.random_split(pd, [trainSize, testSize])

trainDataLoader = DataLoader(trainData, batch_size = 2, shuffle=True)
testDataLoader = DataLoader(testData, batch_size = 2, shuffle=True)
'''

class PosterDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None, genres=None):
        # csv_file: location of csv file
        # img_dir: location of image directory
        
        # transform: some transform object you can use to modify images. not necessary for project IMO
        # target_transform: something about transforming the label? not applicable to us
        
        # Genre: Lst parameter.
        
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((256,256)),  # Resize images to 256/256
        ])
        self.target_transform = target_transform
        self.df = self.df[~np.isnan(self.df["Score"])]
        if genres != None:
            finalBool = 0
            
            # Add columns that tell you whether or not the movie is in the genre
            for genre in genres:
                newCol = self.df['Genre'].str.contains(genre, case=False)
                
                self.df[genre] = newCol
                
                if isinstance(finalBool, int):
                    finalBool = (self.df[genre] == True)
                else:
                    finalBool = finalBool | (self.df[genre] == True)
                    
            # Now the dataframe will only have movies of the genre you asked for.
            self.df = self.df[finalBool]
            self.df = self.df.reset_index(drop=True)
            
        #print(self.df.head())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.img_dir + '/' + str(self.df.iloc[idx]['imdbId']) + '.jpg'
        image = read_image(img_path).to(torch.float)
        label = self.df.iloc[idx]['Score']
        if np.isnan(label):
            raise ArithmeticError(f"Img idx {idx} with imdbID {self.df.iloc[idx]['imdbId']} has invalid score")
        if self.transform:
            image = self.transform(image)
            
        return image, label
