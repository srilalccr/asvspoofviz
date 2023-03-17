import torchaudio
import torchaudio.functional as F
from asv_plot import plot_spectrogram_synthesis

def synthesise_audio_codec(waveform, sample_rate):

    # Apply filtering and change sample rate
    filtered, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
        waveform,
        sample_rate,
        effects=[
            ["lowpass", "4000"],
            [
                "compand",
                "0.02,0.05",
                "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                "-8",
                "-7",
                "0.05",
            ],
            ["rate", "8000"],
        ],
    )
    
        # Apply telephony codec
    codec_applied = F.apply_codec(filtered, sample_rate2, format="gsm")

    plot_spectrogram_synthesis(waveform, sample_rate, codec_applied, sample_rate2, title="GSM Codec Applied")