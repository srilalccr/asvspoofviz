import torch
import torchaudio
from asv_classes import asvDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import fnmatch
import os
from asv_plot import plot_waveform, plot_spectrogram, get_spectrogram, plot_spectrogram_fake_or_real, plot_spectrogram_subband
from synthesise_audio import synthesise_audio_codec

def initializeDataLoader(files_PATH, labels_PATH):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    dataloader_parameters = {'batch_size': 64,
            'shuffle': True,
            'num_workers': 0}
    max_epochs = 100

    
    train_df = pd.read_csv(labels_PATH, sep=" ", header=None,  names=["LA_key", "FileName", "compliance","country", "AttackType", "spoof", "trim", "progress"])

    print("Total audio files:", train_df.groupby("spoof").value_counts())

    train_id, test_id = train_test_split(train_df, shuffle=True, stratify = train_df["spoof"], test_size=0.2)
    #print(len(train_id))
    #print(len(test_id))
    print(len(fnmatch.filter(os.listdir(files_PATH),'*.flac')))

    train_dataset = asvDataset(train_id, root_dir=files_PATH, is_train=True)
    test_dataset = asvDataset(test_id, root_dir=files_PATH, is_train=True)
    print(f"There are {len(train_dataset)} samples in the train dataset")
    print(f"There are {len(test_dataset)} samples in the test dataset")
    #train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_parameters)
    #test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_parameters)
    waveform, samplerate, audio_type = train_dataset[0]
    plot_waveform(waveform, samplerate)
    spectrogram = get_spectrogram(waveform, samplerate)
    plot_spectrogram(torch.abs(spectrogram[0]),waveform, samplerate, title=audio_type, aspect="equal", xmax=spectrogram.size()[2])

    #datsetIterator = iter(train_dataset)
    #waveform, samplerate = next(datsetIterator)
   
    #dataloader_dict = {}
    #dataloader_dict["train"] = train_loader
    #dataloader_dict["test"] = test_loader

    #pipeline_RNNT_ASR(train_dataset)

    # Loop over epochs
   # for epoch in range(max_epochs):
        # Training
     #   print ("Epoch:,", epoch)
        #for waveform, samplerate in next(iter(train_loader)):
        #    pass   
        #print(f"Feature batch shape: {train_features.size()}")
        #print(f"Labels batch shape: {train_labels.size()}")



    #train_features, train_labels = train_loader
    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_labels.size()}")

def initializeFakeOrRealLoader(files_PATH_fake, files_PATH_real):
    waveform_fake, sample_rate_fake = torchaudio.load(files_PATH_fake + "file9989.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav")
    waveform_real, sample_rate_real = torchaudio.load(files_PATH_real + "file999.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav")
    spectrogram_fake = get_spectrogram(waveform_fake, sample_rate_fake)
    spectrogram_real = get_spectrogram(waveform_real, sample_rate_real)
    plot_spectrogram_fake_or_real(torch.abs(spectrogram_fake[0]),waveform_fake, sample_rate_fake,
                                  torch.abs(spectrogram_real[0]),waveform_real, sample_rate_real,
                                    title = "Fake vs Real", aspect="equal", xmax=spectrogram_fake.size()[2])
    plot_spectrogram_subband(waveform_fake.numpy()[0], waveform_real.numpy()[0])
    synthesise_audio_codec(waveform_real, sample_rate_real)
    



    

    
if __name__ == "__main__":
   files_PATH = "./Data/ASVspoof2021_LA_eval/flac/"
   labels_PATH = "./Data/LA-keys-full/keys/LA/CM/trial_metadata.txt"
   
   files_PATH_fake = "./Data/for-2seconds/training/fake/"
   files_PATH_real = "./Data/for-2seconds/training/real/"
   
   initializeDataLoader(files_PATH, labels_PATH)
   initializeFakeOrRealLoader(files_PATH_fake, files_PATH_real)
else:
    print ("ASV spoof visualization imported")
