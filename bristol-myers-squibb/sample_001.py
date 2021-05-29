import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
import fastcore
from fastcore.parallel import parallel


def load_dataset() -> pd.DataFrame:
    train_labels = pd.read_csv('../input/train_labels.csv')
    layers = ['version', 'chemical_notation']
    for layerid, layer in enumerate(layers):
        train_labels[layer] = [(train_label.split('/'))[layerid]
                               for train_label
                               in train_labels['InChI']]
    layers = ['c', 'h', 'b', 't', 'm', 's', 'i']
    for layerid, layer in enumerate(layers):
        train_labels[layer] = [
            ''.join(
                [
                 splitlayer if splitlayer[0] == layer else ''
                     for splitlayer in train_label.split('/')
                ]
            )
            for train_label in train_labels['InChI']
        ]
        train_labels[layer] = ['' if len(item) == 0 else item[1:]
                               for item in train_labels[layer]]

    return train_labels


def show_sampledata(train_labels: pd.DataFrame):
    fig = plt.figure(figsize=(10, 12))
    columns = 3
    rows = 6
    for i, image_id in enumerate(train_labels['image_id'].values[0:18]):
        file_path = f'../input/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png'
        img = Image.open(file_path)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        plt.subplots_adjust(hspace=.5)
        plt.title('/n'.join(wrap(train_labels['InChI'].values[i], 20)))

    plt.show()

    return


def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel)
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)


def characterize_image(image_id):
    file_path = f'../input/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png'
    img = Image.open(file_path)
    bands = img.getbands()
    bands4 = [chan in bands for chan in ['L', 'R', 'G', 'B']]
    imgarr = np.array(img)

    timg = trim(img)
    timgarr = np.array(timg)

    return (img.width, img.height, img.mode, bands4[0], bands4[1],
            bands4[2], bands4[3], np.sum(imgarr<255)/imgarr.size,
            timg.width, timg.height, np.sum(timgarr<255)/timgarr.size)


def main():
    train_labels = load_dataset()
    # show_sampledata(train_labels)

    out1 = parallel(
        characterize_image,
        [image_id for image_id in list(train_labels['image_id'].values)],
        n_workers=4,
        progress=True
    )

    print(out1)



if __name__ == '__main__':
    main()
