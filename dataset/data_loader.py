import torch
import numpy as np
import torch.nn as nn
import torchvision
import soundfile as sf
import torch.nn.functional as F
import librosa


class AudioDataset(nn.Module):
    def __init__(self, partition, instruments, hdf_dir, shapes, random_hops):
        super(AudioDataset, self).__init__()
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")
        self.sr = sr
        self.instruments = instruments
        self.shapes = shapes
        self.random_hops = random_hops

        print('Preparing {} dataset...'.format(partition))
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["length"] for song_idx in range(len(f))]
            lengths = sum(lengths)
        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]
        self.dataset = h5py.File(self.hdf_dir, 'r', driver='core')

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        audio_idx = self.start_pos.bisect_right(item)
        end_pos = start_target_pos + self.shapes["length"]
        if end_pos > audio_length:
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        audio_stft = self.dataset[item]

        audio_stft = np.pad(audio_stft, [(0, 0), (pad_front + 16, pad_back + 16)], mode="constant",
                            constant_values=(0, 0))
        target_acc = self.dataset[str(audio_idx)]["bin_mask_acc"][:, start_pos:end_pos]
        target_vocal = self.dataset[str(audio_idx)]["bin_mask_v
