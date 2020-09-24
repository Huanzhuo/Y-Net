import os
import glob
from tqdm import tqdm
import h5py
import musdb
import numpy as np
import librosa as lib
import soundfile
import museval


INSTUMENTS = ['vocals', "accompaniment", "mix"]
sr = 16000
fft_length = 1022
hop_length = 256
partitions = ['train', 'val', 'test']


def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")


def getMUSDBHQ(database_path):
    subsets = list()

    for subset in ["train", "test"]:
        print("Loading " + subset + " set...")
        tracks = glob.glob(os.path.join(database_path, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            example = dict()
            for stem in ["mix", "bass", "drums", "other", "vocals"]:
                filename = stem
                audio_path = os.path.join(track_folder, filename + ".wav")
                example[stem] = audio_path

            # Add other instruments to form accompaniment
            acc_path = os.path.join(track_folder, "accompaniment.wav")

            if not os.path.exists(acc_path):
                print("Writing accompaniment to " + track_folder)
                stem_audio = []
                for stem in ["bass", "drums", "other"]:
                    audio, sr = load(example[stem], sr=None, mono=False)
                    stem_audio.append(audio)
                acc_audio = np.clip(sum(stem_audio), -1.0, 1.0)
                write_wav(acc_path, acc_audio, sr)

            example["accompaniment"] = acc_path

            samples.append(example)

        subsets.append(samples)
    train_val_list = subsets[0]
    test_list = subsets[1]

    np.random.seed(42)
    train_list = np.random.choice(train_val_list, 75, replace=False)
    val_list = [elem for elem in train_val_list if elem not in train_list]
    dataset = {'train': train_list,
               'val': val_list,
               'test': test_list}
    return dataset


def load(path, sr=sr, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = lib.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr


def nfft(window_size):
    return int(2**np.ceil(int(np.log2(window_size))))


def stft(path):
    sound, _ = lib.load(path, sr=sr, mono=True)
    stft_mat = lib.stft(sound, fft_length, hop_length=hop_length, center=True)
    S = np.abs(stft_mat)
    # log_s = lib.power_to_db(S ** 2, ref=np.max)
    return stft_mat, S


def make_h5py(dataset, h5_dir):
    for partition in partitions:
        hdf_dir = os.path.join(h5_dir, partition + '.hdf5')
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)

        with h5py.File(hdf_dir, "w") as f:
            f.attrs["sr"] = sr
            f.attrs["fft_length"] = fft_length
            f.attrs["hop_length"] = hop_length

            example = dataset[partition][0]
            mix_stft, _ = stft(example["mix"])
            data = np.zeros([mix_stft.shape[0], mix_stft.shape[1], 2], dtype=np.float32)
            bass_stft, _ = stft(example['bass'])
            drum_stft, _ = stft(example['drums'])
            vocal_stft, _ = stft(example['vocals'])
            acc_stft, _ = stft(example['accompaniment'])

            data_bass = np.zeros([bass_stft.shape[0], bass_stft.shape[1], 2], dtype=np.float32)
            data_drum = np.zeros([drum_stft.shape[0], drum_stft.shape[1], 2], dtype=np.float32)
            data_vocal = np.zeros([vocal_stft.shape[0], vocal_stft.shape[1], 2], dtype=np.float32)
            data_acc = np.zeros([acc_stft.shape[0], acc_stft.shape[1], 2], dtype=np.float32)

            start_pos = 0
            for idx, example in enumerate(tqdm(dataset[partition][1:])):
                # Load data
                mix_stft, _ = stft(example["mix"])
                vocal_stft, _ = stft(example["vocals"])
                bass_stft, _ = stft(example["bass"])
                drum_stft, _ = stft(example["drums"])
                acc_stft, _ = stft(example["accompaniment"])

                l = mix_real.shape[1]
                mix = np.concatenate([mix_stft.real[:, :, np.newaxis], mix_stft.imag[:, :, np.newaxis]], axis=2)
                data = np.concatenate([data, mix], axis=1)



                start_pos += l

            # Add to HDF% file
            f.create_dataset(name='data', shape=data.shape, dtype=data.dtype, data=data)
            f.create_dataset(name='bass', shape=data_bass.shape, dtype=data_bass.dtype, data=data_bass)
            f.create_dataset(name='drum', shape=data_drum.shape, dtype=data_drum.dtype, data=data_drum)
            f.create_dataset(name='vocal', shape=data_vocal.shape, dtype=data_vocal.dtype, data=data_vocal)
            f.create_dataset(name='acc', shape=data_acc.shape, dtype=data_acc.dtype, data=data_acc)


if __name__ == '__main__':
    H5_Path = '../../dataset/H5'
    dataset = getMUSDBHQ('../../WaveUNet/musdb18wav')
    make_h5py(dataset, H5_Path)








