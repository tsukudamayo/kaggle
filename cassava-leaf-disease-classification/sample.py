import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2


CFG = {
    'fole_num': 5,
    'seed': 719,
    'model_arch': 'tf_efficient_b4_ns',
    'img_size': 512,
    'epochs': 10,
    'train_bs': 32,
    'valid_bs': 32,
    'lr': 1e-4,
    'num_workers': 4,
    'accum_iter': 1,
    'verbose_step': 1,
    'device': 'cuda:0',
    'tta': 3,
    'used_epochs': [6, 7, 8, 9],
    'weights': [1, 1, 1, 1]
}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    
    return im_rgb


class CassavaDataset(Dataset):
    def __init__(
        self,
        df,
        data_root,
        transform=None,
        output_label=True,
        ):

        super().__init__()
    

    

    


def run():
    train = pd.read_csv('../input/train.csv')
    print(train.head())
    print(train.label.value_counts())
    submission = pd.read_csv('../input/sample_submission.csv')
    print(submission.head())

    img = get_img('../input/train_images/1000015157.jpg')
    plt.imshow(img)
    plt.show()


def main():
    run()
    

if __name__ == '__main__':
    main()
    
