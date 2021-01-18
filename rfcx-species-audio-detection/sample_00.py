from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import albumentations


# class SpeciesModel(tez.Model):
    


class AudioDataset:
    def __init__(
        self,
        audio_paths,
        targets,
        offset,
        duration,
        augumentations=None,
        channel_first=False,
        grayscale=True,
    ):
        self.audio_paths = audio_paths
        self.targets = targets
        self.offset = offset
        self.duration = duration
        self.augumentations = augumentations

    def __len__(self):
        return len(self.audio_paths)

    def __getitem(self, item):
        targets = self.targets[item]
        image = build_spectrogram(
            self.audio_paths[item],
            self.offset[item],
            self.duration[item]
        )
        image = np.array(image)
        if self.augumentations is not None:
            augumented = self.augumentations(image=image)
            image = augumented['image']
        image = np.nan_to_num(image)
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        return {
            'image': image_tensor,
            'targets': torch.tensor(targets, dtype=torch.float)
        }



def mono_to_color(
    X,
    mean=None,
    std=None,
    norm_max=None,
    norm_min=None,
    eps=1e-6,
    ):

    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min

    if (_max - _min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V
    

def build_spectrogram(path, offset, duration=12):
    y, sr = librosa.load(
        path,
        offset=np.floor(offset),
        duration=np.ceil(duration)
    )
    total_secs = y.shape[0] / sr
    M = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
    )
    M = librosa.power_to_db(M)
    M = mono_to_color(M)

    return M



def main():
    run()


def run():
    path = Path('../input/rfcx-species-audio-detection')
    df = pd.read_csv(path/'train_tp.csv')
    files = df.recording_id.tolist()
    fnames = df.recording_id.unique().tolist()
    df_gr = df.groupby(['recording_id'])

    bird_dict = {}
    for fn in tqdm(fnames):
        lbls = np.zeros(24)
        temp = df_gr.get_group(fn)
        sps = temp.species_id.unique()
        for ss in sps:
            lbls[ss] = 1
        bird_dict[fn] = lbls

    bird_df_ = pd.DataFrame.from_dict(
        bird_dict,
        orient='index'
    )
    bird_df = bird_df_.reset_index()
    bird_df.columns = ['recording_id'] + ['species_id_' + str(x) for x in range(24)]

    df_agg = df.groupby(['recording_id'])\
      .agg({
          't_min': lambda x: min(x),
          't_max': lambda x: max(x),
      })\
      .reset_index()

    df_agg['duration'] = df_agg['t_max'] - df_agg['t_min']
    df_agg['duration'] = np.ceil(df_agg['duration'].values)+5

    trn_df = bird_df.merge(df_agg, on='recording_id', how='left')
    trn_df['recording_id'] = \
      '../input/rfcx-species-audio-detection/train/' + \
      trn_df['recording_id'] + \
      '.flac'
    
    tar_cols = ['species_id_' + str(x) for x in range(24)]
    

    # ------------#
    # spectorgram #
    # ------------#
    n = 56
    img = build_spectrogram(
        trn_df.iloc[n]['recording_id'],
        offset=int(trn_df.iloc[n]['t_min']),
        duration=int(trn_df.iloc[n]['duration'])
    )
    plt.figure(1, figsize=(10, 8))
    plt.imshow(img, cmap='inferno')
    plt.show()
    print(img.shape)

    # -------------------- #
    # MultiStraitfiedKfold #
    # -------------------- #
    trn_df = trn_df.sample(
        frac=1.,
        random_state = 2020,
    )
    trn_df['kfold'] = -1
    y = trn_df[tar_cols].values
    kf = MultilabelStratifiedKFold(
        n_splits=5,
        random_state=2020,
        shuffle=True,
    )
    for fold, (trn_, val_) in enumerate(kf.split(X=trn_df, y=y)):
        trn_df.loc[val_, 'kfold'] = fold

    print(trn_df.head())
    fig = sns.violinplot(
        data=trn_df,
        x='duration',
    )
    fig.set_title('duration')
    plt.show()

    # ------------ #
    # load Dataset #
    # ------------ #
    IMAGE_SIZE = 256
    train_aug = albumentations.Compose(
        [
            albumentations.Resize(
                200,
                600,
                p=1.0
            ),
            albumentations.Normalize(
                mean=[0.485],
                std=[0.229],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Resize(
                200,
                600,
                p=1.0,
            ),
            albumentations.Normalize(
                mean=[0.485],
                std=[0.229],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
    )

    FOLD = 2
    df_train, df_valid = trn_df[trn_df.kfold!=FOLD], trn_df[trn_df.kfold==FOLD]
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    train_targets = df_train[tar_cols].values
    valid_targets = df_valid[tar_cols].values

    train_dataset = AudioDataset(
        df_train.recording_id,
        train_targets,
        offset=df_train['t_min'].values,
        duration=df_train['t_min'].values,
        augumentations=train_aug,
    )
    valid_dataset = AudioDataset(
        df_valid.recording_id,
        valid_targets,
        offset=df_train['t_min'].values,
        duration=df_valid.duration.values,
        augumentations=valid_aug,
    )
    plt.figure(1, figsize=(10, 6))
    plt.imshow(
        valid_dataset[5]['image'].numpy()[0,:,:],
        cmap='inferno'
    )
    plt.show()


    

       

    
        
        
            

                    

if __name__ == '__main__':
    main()
