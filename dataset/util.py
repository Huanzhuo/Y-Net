import librosa
import numpy as np
import soundfile


def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")


def load_audio(sr, path):
    return librosa.load(path, sr, mono=True)


def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr


def crop(mix, targets, shapes):
    '''
    Crops target audio to the output shape required by the model given in "shapes"
    '''
    for key in targets.keys():
        if key != "mix":
            targets[key] = targets[key][:, shapes["start_frame"]:shapes["end_frame"]]
    return mix, targets


def random_amplify_raw(mix, targets, min, max):
    '''
    Data augmentation by randomly amplifying sources before adding them to form a new mixture
    :param mix: Original mixture (optional, will be discarded)
    :param targets: Source targets
    :param shapes: Shape dict from model
    :param min: Minimum possible amplification
    :param max: Maximum possible amplification
    :return: New data point as tuple (mix, targets)
    '''
    new_mix = 0
    for key in targets.keys():
        if key == "bass" or "drums" or "other":
            targets[key] = targets[key] * np.random.uniform(min, max)
        if key == 'vocals':
            targets[key] = targets[key] * np.random.uniform(min, max)
    targets["accompaniment"] = targets["bass"] + targets["drums"] + targets["other"]
    new_mix = targets["vocals"] + targets["accompaniment"]
    return new_mix, targets


def random_amplify_speech(mix, targets, min_thres, max_thres, num_chs):
    new_mix = np.zeros_like(mix)
    for i in range(num_chs):
        targets[i, :] = targets[i, :] * np.random.uniform(min_thres, max_thres)
    new_mix[0, :] = np.sum(targets, axis=0)
    return new_mix, targets


