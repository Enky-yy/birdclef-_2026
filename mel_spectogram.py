import librosa

def audio_to_mel(path):
    y,sr = librosa.load(path, sr=32000)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels =128,
        fmin = 20,
        fmax = 16000
    )

    mel = librosa.power_to_db(mel)

    return mel