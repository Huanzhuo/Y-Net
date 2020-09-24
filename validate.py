import time
from tqdm import tqdm
import soundfile as sf
import os
import numpy as np
import librosa
from torch.utils.data import DataLoader
from dataset.audioDatasetHdf import AudioDataset
import museval
import torch
from models.y_net import Y_Net
from dataset.util import *
from apex import amp
from losses.conv_tasnet_loss import calculate_loss

# MODEL_PATH = '../y-net-result/2020-07-24-00/model_best.pt'
MODEL_PATH = '../y-net-result/2020-07-30-11/model_best.pt'
shapes = {'length': 32000}


def valiadate_folds(args, song_list, source_path, target_name=None):
    """
    Iteration every song from a existing folder
    :param args:
    :param song_list:
    :param source_path:
    :param target_name:
    :return: sdr for every song
    """

    if target_name is None:
        target_name = ["accompaniment"]
    if args.mode == 'align':
        output_length = shapes_align["length"]
    elif args.mode == 'misalign':
        output_length = shapes["output_length"]
    sdrs = {}

    for name in song_list:
        song_path = source_path + name + '/mix.wav'
        results = validate(args, song_path, output_length, target_name)
        sdrs[name] = cal_sdr_matrix(source_path + name, results, output_length, target_name)
        if args.save_result:
            save_results(name, results, args.result_path)
    print(sdrs)


def validate(song_path):
    # Read song
    mix, _ = librosa.load(song_path, sr=16000, mono=True)
    length = mix.shape[0]

    # Create Model and Load parameters
    print("Creating and Loading model")
    # state_dict = torch.load(MODEL_PATH, map_location='cpu')
    # state_pa = state_dict['model_state_dict'].float()
    # model = Y_Net()
    # model, _ = amp.initialize(model, opt_level="O1")
    # model.load_state_dict(model.load_stat e_dict(torch.load(MODEL_PATH, map_location='cpu ')))
    model = torch.load(MODEL_PATH)['model']
    # model = Y_Net()
    # model = model.cuda()
    # state_dict = torch.load(MODEL_PATH, map_location='cpu')
    # state_pa = state_dict['model_state_dict']
    # model.load_state_dict(state_pa)
    # model = amp.initialize(model, opt_level="O1")
    model.eval()
    # model = model.cuda()

    if length % shapes['length'] != 0:
        pad_back = shapes['length'] - length % shapes['length']
    else:
        pad_back = 0

    mix = np.concatenate([mix, np.zeros(pad_back, dtype=np.float32)])
    output_frames = mix.shape[0]

    mix = torch.tensor(mix).view(1, -1)
    vocals = np.zeros(output_frames, dtype=np.float32)
    bass = np.zeros(output_frames, dtype=np.float32)
    drums = np.zeros(output_frames, dtype=np.float32)
    others = np.zeros(output_frames, dtype=np.float32)
    acc = np.zeros(output_frames, dtype=np.float32)

    for start_frame in tqdm(range(0, output_frames, shapes['length'])):
        end_frame = start_frame + shapes['length']
        x = mix[..., start_frame:end_frame]
        x= x.cuda()
        # if x.max() < 0.1:
        #     mask = torch.tensor([0])
        # else:
        #     mask = torch.tensor([1])
        output = model(x)
        # output = output * mask
        # bs = output[0][0, 0, :].detach().cpu().numpy()
        # dm = output[0][0, 1, :].detach().cpu().numpy()
        # ot = output[0][0, 2, :].detach().cpu().numpy()
        # vo = output[0][0, 3, :].detach().cpu().numpy()

        # acc = output[0, 0, :].detach().numpy()
        # bass[start_frame: start_frame + shapes['length']] = bs
        # drums[start_frame: start_frame + shapes['length']] = dm
        # others[start_frame: start_frame + shapes['length']] = ot
        # vocals[start_frame: start_frame + shapes['length']] = vo

        # vo = output[0][0, 0, :].detach().cpu().numpy()
        ac = output[0][0, 1, :].detach().cpu().numpy()
        vo = x.detach().cpu().numpy() - ac
        vocals[start_frame: start_frame + shapes['length']] = vo
        acc[start_frame: start_frame + shapes['length']] = ac
    soundfile.write('./vocals.wav', vocals, samplerate=16000)
    soundfile.write('./acc.wav', acc, samplerate=16000)
    print('Finish...')


def validate_center():
    pass


def cal_sdr_matrix(sources_path, results, output_length, target_name):
    sdrs = {}
    for name in target_name:
        source_path = sources_path + '/' + name + '.wav'
        raw_data = librosa.load(source_path, sr=args.sr, mono=True)
        raw_data = padding(raw_data, output_length)
        result = results[name]
        raw_data = raw_data[np.newaxis, :, np.newaxis]
        result = result[np.newaxis, :, np.newaxis]
        assert raw_data.shape == result.shape
        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(raw_data, result)
        sdrs[name] = sdr
    return sdrs


def padding(raw_data, output_length):
    length = raw_data.shape[0]
    if length % output_length != 0:
        pad_back = output_length - length % output_length
    else:
        pad_back = 0
    raw_data = np.concatenate([raw_data, np.zeros(pad_back, dtype=np.float32)])
    return raw_data


def cal_sdr(ori_path, path, shapes):
    raw_data, _ = librosa.load(ori_path, sr=16000, mono=True)
    data_, _ = librosa.load(path, sr=16000, mono=True)
    length = raw_data.shape[0]

    if length % shapes['length'] != 0:
        pad_back = shapes['length'] - length % shapes['length']
    else:
        pad_back = 0

    raw_data = np.concatenate([raw_data, np.zeros(pad_back, dtype=np.float32)])
    raw_data = raw_data[np.newaxis, :, np.newaxis]
    data_ = data_[np.newaxis, :, np.newaxis]

    assert raw_data.shape[0] == data_.shape[0]
    sdr, isr, sir, sar, _ = museval.metrics.bss_eval(raw_data, data_)
    sdr = sdr[sdr > 0]
    print(np.nanmean(sdr))
    print('test')


if __name__ == '__main__':
    song_path = '../WaveUNet/musdb18wav/test/Cristina Vane - So Easy/mix.wav'
    # song_path = '../music/jay.wav'
    validate(song_path)
    path = './vocals.wav'
    raw_path = '../WaveUNet/musdb18wavdown/test/Cristina Vane - So Easy/vocals.wav'
    # cal_sdr(raw_path, path, shapes)
