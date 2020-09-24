import os
import torch
import torch.nn as nn
from dataset.util import *


class AudioDataset(nn.Module):
    def __init__(self, sr, path_list, target_length, name_list=None, random_hops=False, audio_transform=None):
        super(AudioDataset, self).__init__()
        if name_list is None:
            name_list = ['s1.wav', 's2.wav', 'mix.wav']
        self.sr = sr
        self.path_list = path_list
        self.target_length = target_length
        self.name_list = name_list
        self.random_hops = random_hops
        self.num_chs = len(name_list) - 1
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        audio, targets =  self.load_t_series(idx)
        return torch.tensor(audio).squeeze(), torch.tensor(targets)

    def get_path(self, idx):
        path_dict = {}
        for name in self.name_list:
            path = self.path_list[idx] + '/' + name
            name_ = name.split('.')[0]
            path_dict[name_] = path
        return path_dict


    def load_t_series(self, idx):
        sr = self.sr
        path_dict = self.get_path(idx)
        data_dict = {}
        source_length = 0
        for name in path_dict.keys():
            if name == 'mix':
                source_mix, _ = load_audio(sr, path_dict[name])
                data_dict[name] = source_mix
                source_length = source_mix.shape[0]
            else:
                sep, _ = load_audio(sr, path_dict[name])
                data_dict[name] = sep

        audio, targets = self.resize_audio(data_dict, source_length)
        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)
        return audio, targets

    def resize_audio(self, audio_dict, source_length):
        audio = np.zeros([1, self.target_length], dtype=np.float32)
        targets = np.zeros([self.num_chs, self.target_length], dtype=np.float32)

        if self.random_hops:
            start_pos = np.random.randint(0, max(1, source_length - self.target_length + 1))
        else:
            start_pos = 0

        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        end_pos = start_pos + self.target_length
        if end_pos > source_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - source_length
            end_pos = source_length
        else:
            pad_back = 0

        i = 0
        for name in audio_dict.keys():
            if pad_back > 0 or pad_front > 0:
                audio_dict[name] = np.pad(audio_dict[name], (pad_front, pad_back), mode='constant', constant_values=0.0)
                if name == 'mix':
                    audio[0, :] = audio_dict['mix'][:]
                else:
                    targets[i, :] = audio_dict[name][:]
                    i += 1
                continue

            if name == 'mix':
                audio[0, :] = audio_dict['mix'][start_pos:end_pos]
            else:
                targets[i, :] = audio_dict[name][start_pos:end_pos]
                i += 1

        return audio, targets


if __name__ == '__main__':
    path = '/home/hejia/Dataset/libDataset/dev/'
    path_list = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            path_list.append(root + dir)

    augment_func = lambda mix, targets: random_amplify_speech(mix, targets, 0.7, 1.0, 2)
    dataset = AudioDataset(8000, path_list, 15996, random_hops=True, audio_transform=augment_func)
    for i in range(len(dataset)):
        audio, targets = dataset[i]
    print('test')




