import torch
import torchaudio
import os

class asvDataset(torch.utils.data.Dataset):

  def __init__(self, dataframe, root_dir, is_train = False, transform = None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform

  def __len__(self):
        return len(self.dataframe)

  def __getitem__(self, index):
        # Select sample
        if torch.is_tensor(index):
             index = index.tolist()

        # Load data and get label
        if (self.is_train):
            print(self.root_dir,self.dataframe.iloc[index, 1],'.flac')
            waveform, sample_rate = torchaudio.load(self.root_dir +  self.dataframe.iloc[index, 1] + '.flac')
            #x = torch.load( self.root_dir +  self.dataframe.iloc[index, 1] + '.flac') #audio
            #y = torch.tensor(self.dataframe.iloc[index, 5]) #label
        else:
            # y=torch.tensor(1) 
            pass

        if self.transform:
             pass

        return waveform, sample_rate