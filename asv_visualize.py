import torch
from asv_classes import asvDataset
import pandas as pd
from sklearn.model_selection import train_test_split

def initializeDataLoader():

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    dataloader_parameters = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 6}
    max_epochs = 100

    train_df = pd.read_csv("./data/ASVspoof2021.LA.cm.train.trn.txt", sep=" ", header=None,  names=["LA_key", "FileName", "compliance","country", "AttackType", "spoof", "trim", "progress"])

    #print(train_df.groupby("spoof").value_counts())

    train_id, test_id = train_test_split(train_df, shuffle=True, stratify = train_df["spoof"], test_size=0.2)
    #print(len(train_id))
    #print(len(test_id))

    train_dataset = asvDataset(train_id, root_dir="./data/flac/", is_train=True)
    test_dataset = asvDataset(test_id, root_dir="./data/flac/", is_train=True)
    #print(train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_parameters)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_parameters)
    #dataloader_dict = {}
    #dataloader_dict["train"] = train_loader
    #dataloader_dict["test"] = test_loader

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for waveform, samplerate in train_loader:
            pass   
        #print(f"Feature batch shape: {train_features.size()}")
        #print(f"Labels batch shape: {train_labels.size()}")




    #train_features, train_labels = train_loader
    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_labels.size()}")

    

    

initializeDataLoader()