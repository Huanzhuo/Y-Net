[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Y-Net

A dual path convolutional neural network model for high accuracy Blind Source Separation (BSS).

# Usage

## Train Y-net

1. Downloading MUSDB18 dataset (we recommend you to download .wav version or transform the mp4 to .wav format).
2. Converting the wav files to h5 files: <code>python ./dataset/make_dataset</code>, to speed up training.
3. Training a Y-Net model: <code>python train.py</code>.
4. Modifying the structure of the Y-Net model at <code>./configs/defaults.py</code>.
5. Modifying the function <code>train_cfg()</code> at <code>./train.py</code> to change the hyperparameters of the Y-Net model.
6. Defying the target output sources at <code>./train.py INSTRUMENTS</code>.
7. Modifying the path to validate the Y-Net model: <code>python validate.py</code>.

## Separation Accuracy 

# Citation

If you like our repository, please cite our papers.

``` 
@INPROCEEDINGS{Wu2012:Y,
    AUTHOR={Huanzhuo Wu and Jia He and M{\'a}t{\'e {T{\"o}m{\"o}sk{\"o}zi} and Frank H.P. Fitzek},
    TITLE="{Y-Net:} A Dual Path Model for High Accuracy Blind Source Separation",
    BOOKTITLE="2020 IEEE Globecom Workshops (GC Wkshps): IEEE GLOBECOM 2020 Workshop on  Future of Wireless Access for Industrial IoT (FutureIIoT) (GC 2020 Workshop - FIIoT)",
    ADDRESS="Taipei, Taiwan",
    DAYS=6,
    MONTH=dec,
    YEAR=2020
}
```

# About Us

We are researchers at the Deutsche Telekom Chair of Communication Networks (ComNets) at TU Dresden, Germany. Our focus is on in-network computing.

* **Huanzhuo Wu** - huanzhuo.wu@tu-dresden.de
* **Jia He** - jia.he@mailbox.tu-dresden.de

# License

This project is licensed under the [MIT license](./LICENSE).
