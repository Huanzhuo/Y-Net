import os
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from math import floor
from apex import amp
import librosa
from contextlib import redirect_stdout
from configs import cfg
from dataset.util import *
import torch.optim as optim
from models.y_net import Y_Net
from losses.conv_tasnet_loss import calculate_loss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.audioDatasetHdf import AudioDataset
from torch.utils.tensorboard import SummaryWriter


nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
result_dir = '../y-net-result/{}'.format(nowTime)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def save_model(model, optimizer, step, path):
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model': model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        # 'amp': amp.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        checkpoint['optimizer_state_dict']['param_groups'][0]['lr'] = 0.001
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return step


def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=True,
                        help='use gpu, default True')
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--output_size', type=float, default=2.0,
                        help='Output duration')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sampling rate')
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--channels', type=int, default=1,
                        help="Input channel, mono or sterno, default mono")
    parser.add_argument('--h5_dir', type=str, default='../WaveUNet/H5/',
                        help="Path of hdf5 file")
    parser.add_argument('--val_song', type=str, default='../WaveUNet/musdb18wavdown/test/Cristina Vane - So Easy/mix.wav',
                        help="validate song")
    parser.add_argument('--load_model', type=str, default='../y-net-result/2020-07-23-18/model_best.pth',
                        help="Path of hdf5 file")
    parser.add_argument("--load", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs of half lr")
    parser.add_argument("--hold_step", type=int, default=20,
                        help="Epochs of hold step")
    parser.add_argument("--example_freq", type=int, default=200,
                        help="write an audio summary into Tensorboard logs")
    parser.add_argument("--alpha_1", type=float, default=1.0,
                        help="value of alpha_1")
    parser.add_argument("--alpha_2", type=float, default=0.5,
                        help='value of alpha_2')
    parser.add_argument("--reconst_loss", type=bool, default=False,
                        help='use reconstruction loss or not')
    return parser.parse_args()


def valiadate(model, criterion, val_loader, state, writer, args):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for i, (x, targets, masks) in enumerate(val_loader):
            x = x.cuda()
            targets = targets.cuda()
            masks = masks.cuda()
            output = model(x)
            loss = criterion(output, targets, x, masks, reconst=args.reconst_loss)
            '''
            if (i+1) % 150 == 0:
                writer.add_audio("predicted_bass_{}".format(i), output[0,0,:], state["epochs"], sample_rate=16000)
                writer.add_audio("predicted_drum_{}".format(i), output[0,1,:], state["epochs"], sample_rate=16000)
                writer.add_audio("predicted_others_{}".format(i), output[0,2,:], state["epochs"], sample_rate=16000)
                writer.add_audio("predicted_vocals_{}".format(i), output[0,3,:], state["epochs"], sample_rate=16000)
                writer.add_audio("mix_{}".format(i), x[0,:], state["epochs"], sample_rate=16000)
            '''

            # loss, _, _ = criterion(targets, output)
            total_loss += loss.item()
    return total_loss


def main():
    # record the configs of the model
    yaml_path = result_dir + '/configs.yml'
    print("The configs of Y-Net is followings...")
    with open(yaml_path, 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())
    # generate train configs
    args = train_cfg()
    print(args)

    # generate the summarywriter, dataset and dataloader
    writer = SummaryWriter(result_dir)
    shapes = {'length': 32000}
    args.load = False

    INSTRUMENTS = {"bass": False,
                   "drums": False,
                   "other": False,
                   "vocals": True,
                   "accompaniment": True}
    augment_func = lambda mix, targets: random_amplify_raw(mix, targets, 0.7, 1.0)
    # crop_func = lambda mix, targets: crop(mix, targets, shapes)
    train_dataset = AudioDataset('train', INSTRUMENTS, args.sr, 2, True, args.h5_dir, shapes, augment_func, switch_augment=False)
    val_dataset = AudioDataset('val', INSTRUMENTS, args.sr, 2, False, args.h5_dir, shapes, switch_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # generate the model
    args.load = False
    if args.load:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        model = torch.load('../y-net-result/2020-07-29-21/model_best.pt')['model']
    else:
        model = Y_Net()
        model = model.cuda()
    # generate optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    clr = lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=1e-5)
    # use mix precision training, may reduce the accuracy but increase the training speed
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # generate the loss
    criterion = calculate_loss

    # Set up training state dict that will also be saved into checkpoints
    state = {"worse_epochs": 0,
             "epochs": 0,
             "best_loss": np.Inf,
             'step': 0,
             'shapes': shapes}

    print('Start training...')
    for i in range(args.epochs):
        print("Training one epoch from iteration " + str(state["epochs"]))
        model.train()
        train_loss = 0.0
        for i, (x, targets, masks) in enumerate(train_loader):
            x = x.cuda()
            targets = targets.cuda()
            masks = masks.cuda()
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            writer.add_scalar("learning_rate", cur_lr, state['step'])
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, targets, x, masks)
            train_loss += loss.item()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizer.step()
            # clr.step()
            state['step'] += 1

            print("{:4d}/{:4d} --- Loss: {:.6f} with learnig rate {:.6f}".format(
                i, len(train_dataset) // args.batch_size, loss.item(), cur_lr))

        clr.step()
        val_loss = valiadate(model, criterion, val_loader, state, writer, args)
        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)

        print("Validation loss" + str(val_loss))
        writer.add_scalar("train_loss", train_loss, state['epochs'])
        writer.add_scalar("val_loss", val_loss, state['epochs'])

        # EARLY STOPPING CHECK
        checkpoint_path = args.model_path + str(state['epochs']) + '.pth'
        print("Saving model...")
        if val_loss >= state["best_loss"]:
            state["worse_epochs"] += 1
        else:
            print("MODEL IMPROVED ON VALIDATION SET!")
            state["worse_epochs"] = 0
            state["best_loss"] = val_loss
            state["best_checkpoint"] = checkpoint_path
            best_checkpoint_path = args.model_path + 'best.pt'
            best_state_dict_path = args.model_path + 'best_state_dict.pt'
            save_model(model, optimizer, state, best_checkpoint_path)
            torch.save(model.state_dict(), best_state_dict_path)
        print(state)
        state["epochs"] += 1
        if state["worse_epochs"] > args.hold_step:
            break
    last_model = args.model_path + 'last_model.pt'
    save_model(model, optimizer, state, last_model)
    print("Training finished")
    writer.close()


if __name__ == '__main__':
    main()
