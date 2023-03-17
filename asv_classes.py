import torch
import torchaudio

class asvDataset(torch.utils.data.Dataset):

  def __init__(self, dataframe, root_dir, is_train = False, transform = None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        #print(self.root_dir,self.dataframe.iloc[0, 1],'.flac')

  def __len__(self):
        return len(self.dataframe)

  def __getitem__(self, index):
        # Select sample
        if torch.is_tensor(index):
             index = index.tolist()

        # Load data and get label
        if (self.is_train):
            waveform, sample_rate = torchaudio.load(self.root_dir +  self.dataframe.iloc[index, 1] + '.flac')
            audio_type = (f"Sample {index} is {self.dataframe.iloc[index, 5]} of attack type {self.dataframe.iloc[index, 4]}")
            print(audio_type)
            #x = torch.load( self.root_dir +  self.dataframe.iloc[index, 1] + '.flac') #audio
            #y = torch.tensor(self.dataframe.iloc[index, 5]) #label
        else:
            # y=torch.tensor(1) 
            pass

        if self.transform:
             pass

        return waveform, sample_rate, audio_type
  
  class fake_or_real_Dataset(torch.utils.data.Dataset):

      def __init__(self, root_dir, is_train = False, transform = None):
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
                  waveform, sample_rate = torchaudio.load(self.root_dir +  self.dataframe.iloc[index, 1] + '.flac')
                  audio_type = (f"Sample {index} is {self.dataframe.iloc[index, 5]} of attack type {self.dataframe.iloc[index, 4]}")
                  print(audio_type)
                  #x = torch.load( self.root_dir +  self.dataframe.iloc[index, 1] + '.flac') #audio
                  #y = torch.tensor(self.dataframe.iloc[index, 5]) #label
            else:
                  # y=torch.tensor(1) 
                  pass

            if self.transform:
                  pass

            return waveform, sample_rate, audio_type