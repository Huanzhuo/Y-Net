import musdb
import librosa
import soundfile
import os
import librosa as lib
from tqdm import tqdm
import numpy as np
import torch
from sortedcontainers import SortedList
import h5py
import torch.nn as nn
from dataset.util import *
import random


def getMUSDB(database_path):
    # 导入数据
    mus = musdb.DB(root=database_path, is_wav=False)

    subsets = list()
    for subset in ["train", "test"]:
        tracks = mus.load_mus_tracks(subset)
        samples = list()

        # Go through tracks
        for track in tracks:
            # Skip track if mixture is already written, assuming this track is done already
            # track_path = track.path[:-4]
            track_path = SAVE_PATH + subset + '/' + track.name
            if not os.path.exists(track_path):
                os.mkdir(track_path)
            mix_path = track_path + "/mix.wav"
            acc_path = track_path + "/accompaniment.wav"
            if os.path.exists(mix_path):
                print("WARNING: Skipping track " + mix_path + " since it exists already")

                # Add paths and then skip
                paths = {"mix": mix_path, "accompaniment": acc_path}
                paths.update({key: track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

                samples.append(paths)

                continue

            rate = track.rate

            # Go through each instrument
            paths = dict()
            stem_audio = dict()
            for stem in ["bass", "drums", "other", "vocals"]:
                path = track_path + '/' + stem + ".wav"
                audio = track.targets[stem].audio.T
                soundfile.write(path, audio, rate, "PCM_16")
                stem_audio[stem] = audio
                paths[stem] = path

            # Add other instruments to form accompaniment
            acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
            soundfile.write(acc_path, acc_audio, rate, "PCM_16")
            paths["accompaniment"] = acc_path

            # Create mixture
            mix_audio = track.audio.T
            soundfile.write(mix_path, mix_audio, rate, "PCM_16")
            paths["mix"] = mix_path

            diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
            print("Maximum absolute deviation from source additivity constraint: " + str(
                np.max(diff_signal)))  # Check if acc+vocals=mix
            print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

            samples.append(paths)

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


class AudioDataset(nn.Module):
    def __init__(self, partition, instruments, sr, out_channels, random_hops, hdf_dir, shapes, audio_transform=None, in_memory=False, switch_augment=True, threshold=0.5):
        super(AudioDataset, self).__init__()
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
        self.random_hops = random_hops
        self.sr = sr
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments
        self.shapes = shapes
        self.out_channels = out_channels
        self.switch = switch_augment
        self.threshold = threshold

        print('Preparing {} dataset...'.format(partition))

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes['length']) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]
        self.dataset = h5py.File(self.hdf_dir, 'r', driver="core")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.switch:
            audio, sources = self.switch_getitem(idx)
        else:
            audio, sources = self.pick_one_segment(idx)

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, sources = self.audio_transform(audio, sources)
        idx_temp = 0
        targets = np.zeros([self.out_channels, self.shapes['length']], dtype=np.float32)
        masks = np.ones([self.out_channels], dtype=np.float32)
        if self.out_channels == 1:
            targets = sources['vocals']
        else:
            for k, inst in enumerate(self.instruments.keys()):
                # if inst == 'other':
                #     continue
                if self.instruments[inst]:
                    targets[idx_temp] = sources[inst]
                    if np.max(sources[inst]) < 0.05:
                        masks[idx_temp] = 0
                    idx_temp += 1
        return torch.tensor(audio).squeeze(), torch.tensor(targets), torch.tensor(masks)

    def switch_getitem(self, idx):
        # random pick another
        p = random.random()
        if p > self.threshold:
            indices = [idx] + [random.randint(0, self.length-1)]
            mix_0, sources_0 = self.pick_one_segment(indices[0])
            mix_1, sources_1 = self.pick_one_segment(indices[1])
            mix = np.zeros_like(mix_0, dtype=mix_0.dtype)
            sources = {}
            for k in sources_0.keys():
                if random.random() > 0.5:
                    s = sources_0[k]
                else:
                    s = sources_1[k]
                mix += s
                sources[k] = s
            return mix, sources
        else:
            return self.pick_one_segment(idx)

    def pick_one_segment(self, idx):
        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(idx)

        if audio_idx > 0:
            idx = idx - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.dataset[str(audio_idx)].attrs["length"]
        target_length = self.dataset[str(audio_idx)].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes['length'] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = idx * self.shapes['length']
        start_pos = start_target_pos
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0
        end_pos = start_target_pos + self.shapes['length']
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)
        targets = self.dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        sources = {}
        for id, inst in enumerate(self.instruments.keys()):
            sources[inst] = targets[id:id + 1]
        del targets
        return audio, sources

    def debug(self, idx):
        if self.switch:
            audio, sources, mask = self.switch_getitem(idx)
        else:
            audio, sources, mask = self.pick_one_segment(idx)

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, sources = self.audio_transform(audio, sources)
        idx_temp = 0
        targets = np.zeros([self.out_channels, self.shapes['length']], dtype=np.float32)
        if self.out_channels == 1:
            targets = sources['vocals']
        else:
            for k, inst in enumerate(self.instruments.keys()):
                # if inst == 'other':
                #     continue
                if self.instruments[inst]:
                    targets[idx_temp] = sources[inst]
                    idx_temp += 1
        librosa.output.write_wav('mix.wav', audio[0], 16000)
        librosa.output.write_wav('bass.wav', targets[0], 16000)
        librosa.output.write_wav('drums.wav', targets[1], 16000)
        librosa.output.write_wav('other.wav', targets[2], 16000)
        librosa.output.write_wav('vocals.wav', targets[3], 16000)


if __name__ == '__main__':
    partition = 'val'
    INSTRUMENTS = {"bass": True,
                   "drums": True,
                   "other": True,
                   "vocals": True,
                   "accompaniment": False}
    # shapes = {'length': 320000}
    shapes = {'length': 320000}
    h5_dir = '../../WaveUNet/H5/'
    augment_func = lambda mix, targets: random_amplify_raw(mix, targets, 0.7, 1.0)
    # crop_func = lambda mix, targets: crop(mix, targets, shapes)
    train_dataset = AudioDataset('train', INSTRUMENTS, 16000, 4, True, h5_dir, shapes,
                                 augment_func, switch_augment=False)
    for i in range(len(train_dataset)):
        audio, targets, mask = train_dataset[i]
    print('test')












