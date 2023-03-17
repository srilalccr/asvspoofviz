import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import torchaudio.transforms as torch_transform
import librosa
import torchaudio.functional as torch_functional
import numpy as np

def plot_waveform(waveform, sample_rate, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    print("Saving audio csv")
    #plt.show(block=False)
    #plt.savefig("spoof_audio.svg")
    df = pd.DataFrame(time_axis, columns=['Time'])
    for c in range(num_channels):
        df_channels = pd.DataFrame(waveform[c], columns=[f"Channel{c+1}"])
        df = df.join(df_channels)
    print(df.head(2)) 
    os.makedirs('asvspoofviz-main', exist_ok=True)  
    df.to_csv("asvspoofviz-main/asv_waveform.csv",index=False)

def get_spectrogram(
        waveform, sample_rate,
        n_fft = 400,
        win_len = None,
        hop_len = None,
        power = 2.0,
    
):
    waveform_n = waveform.numpy()
    num_channels, num_frames = waveform_n.shape
    spectrogram = torch_transform.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    spec = spectrogram(waveform)
    print(f"Spectrogram Dimension {n_fft/2+1} frequencies across {spec.size()[2]} time instants tensor {spec.shape}")
    return (spec)

def plot_spectrogram(spec, waveform, sample_rate, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(6, 6))
    axs[0].set_title(f"{title} Spectrogram (db)")
    axs[0].set_ylabel(ylabel)
    axs[0].set_xlabel("frame")
    im = axs[0].imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs[0].set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    #plt.subplot(2,1, 2)
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    axs[1].plot(time_axis, waveform[0], linewidth=1)
    plt.savefig("./asvspoofviz-main/spectrogram.svg", dpi=150)

def plot_spectrogram_synthesis( waveform, sample_rate, title="Spectrogram synthesised", xlim=None):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.savefig("./asvspoofviz-main/spectrogram_synthesised.svg", dpi=150)

def plot_spectrogram_fake_or_real(spec_fake, waveform_fake, sample_rate_fake,
                                  spec_real, waveform_real,sample_rate_real, 
                                   title=None, ylabel=None, aspect="auto", xmax=None):
                                   
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(10, 6))
    axs[0,0].set_title(f" Synthetic (db)")
    axs[0,0].set_ylabel(ylabel)
    axs[0,0].set_xlabel("frame")
    im = axs[0,0].imshow(librosa.power_to_db(spec_fake), origin="lower", aspect=aspect)
    if xmax:
        axs[0,0].set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)

    axs[0,1].set_title(f"Original (db)")
    axs[0,1].set_ylabel(ylabel)
    axs[0,1].set_xlabel("frame")
    im = axs[0,1].imshow(librosa.power_to_db(spec_real), origin="lower", aspect=aspect)
    if xmax:
        axs[0,1].set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)

    #plt.subplot(2,2, 2)
    waveform_fake = waveform_fake.numpy()

    num_channels, num_frames = waveform_fake.shape
    time_axis = torch.arange(0, num_frames) / sample_rate_fake
    axs[1,0].plot(time_axis, waveform_fake[0], linewidth=1)
    plt.savefig("./asvspoofviz-main/spectrogram.svg", dpi=150)

    #plt.subplot(2,2,3)
    waveform_real = waveform_real.numpy()

    num_channels, num_frames = waveform_real.shape
    time_axis = torch.arange(0, num_frames) / sample_rate_real
    axs[1,1].plot(time_axis, waveform_real[0], linewidth=1)
    plt.savefig("./asvspoofviz-main/spectrogram_fake_real.svg", dpi=150)


def plot_spectrogram_subband(waveform_fake, waveform_real, 
                                   title=None, ylabel=None, aspect="auto", xmax=None):
    color = "viridis_r"

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(10, 6))
    
    axs[0,0].set_title(f" Synthetic (db)")
    axs[0,0].set_ylabel("STFT")
    spec = np.abs(librosa.stft(waveform_fake, center=False))
    im = axs[0,0].imshow(librosa.amplitude_to_db(spec, ref=np.max), origin="lower", aspect=aspect, cmap=color)
    if xmax:
        axs[0,0].set_xlim((0, xmax))

    axs[0,1].set_title(f"Original (db)")
    axs[0,1].set_xlabel("time")
    spec = np.abs(librosa.stft(waveform_real, center=False))
    im = axs[0,1].imshow(librosa.amplitude_to_db(spec, ref=np.max), origin="lower", aspect=aspect,cmap=color)
    if xmax:
        axs[0,1].set_xlim((0, xmax))

    axs[1,0].set_ylabel("Constant Q")
    spec = np.abs(librosa.hybrid_cqt(waveform_fake, fmin=15, bins_per_octave=96))
    im = axs[1,0].imshow(librosa.amplitude_to_db(spec, ref=np.max), origin="lower", aspect=aspect,cmap=color)
    if xmax:
        axs[1,0].set_xlim((0, xmax))


    axs[1,1].set_xlabel("time")
    spec = np.abs(librosa.hybrid_cqt(waveform_fake, fmin=15, bins_per_octave=96))
    im = axs[1,1].imshow(librosa.amplitude_to_db(spec, ref=np.max), origin="lower", aspect=aspect,cmap=color)
    if xmax:
        axs[1,1].set_xlim((0, xmax))
    fig.colorbar(im, ax=axs, format="%+2.0f dB")
    plt.savefig("./asvspoofviz-main/subbands.svg", dpi=150)



    


