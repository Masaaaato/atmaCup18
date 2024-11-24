"""
## bacboneの特徴量に走行時特徴量をconcatした後にMLPでregressionするモデル
**注意**
- 画像のpathは適宜変更してください
- 初期検討で, 全ての重みを学習し始めるとスコアが悪かったので走行時特徴量に関連した層のみをfirst stageとして学習していますが, 本当に重要かは検討不十分です.
- CV splittingについて, 走行時特徴量を用いたKmeansクラスタリングをラベル, scene_idをグループとしてStratifiedGroupKFoldを使用しています. 適切に変更してください.
"""

import os
import gc
import math
import random
import argparse
from argparse import Namespace
from joblib import Parallel, delayed
from functools import partial

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import cv2
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


###################
### Utils
###################
# Any commandline arguments can be passed to the script using the `--` prefix
def convert_string(value):
    # Try converting to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try converting to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try converting to bool
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    
    if value is None or value == 'None':
        return None
    
    # Return as string if no other type matches
    return value

# https://www.guruguru.science/competitions/25/discussions/b75b30bb-abcd-482d-b43a-fa325748e48d/
def camera_to_image(P_camera, intrinsic_matrix):
    """前方または後方のカメラ座標系から画像座標系に変換する関数"""
    P_image_homogeneous = np.dot(intrinsic_matrix, P_camera)

    P_image = P_image_homogeneous[:2] / P_image_homogeneous[2]
    return P_image

def image_to_camera(P_image, inverse_intrinsic_matrix, move_x):
    """画像座標系からカメラ座標系に変換する関数"""
    P_image_homogeneous = np.array([P_image[0], P_image[1], 1])  # 同次座標系に変換
    P_camera_homogeneous = np.dot(inverse_intrinsic_matrix, P_image_homogeneous)

    P_camera = P_camera_homogeneous * move_x
    return P_camera

def project_trajectory_to_image_coordinate_system(trajectory: np.ndarray, intrinsic_matrix: np.ndarray):
    """車両中心座標系で表現されたtrajectoryをカメラ座標系に投影する"""
    # カメラの設置されている高さ(1.22m)まで座標系をズラす
    trajectory_with_offset = trajectory.copy()
    trajectory_with_offset[:, 2] = trajectory_with_offset[:, 2] + 1.22

    # 座標の取り方を変更する
    road_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    trajectory_camera = trajectory_with_offset @ road_to_camera # [-y, z, x]

    trajectory_image = np.array([camera_to_image(p, intrinsic_matrix) if p[2] > 0 else np.array([0., 0.]) for p in trajectory_camera])
    return trajectory_image, trajectory_camera[:, 2]

def camera_to_vehicle_center(P_camera, road_to_camera):
    """カメラ座標系から車両の中心座標系に変換する関数"""
    # 逆回転行列を使ってカメラ座標系から車両中心座標系に変換
    camera_to_road = np.linalg.inv(road_to_camera)
    P_vehicle_center = np.dot(P_camera, camera_to_road)
    # 高さオフセットを削除
    P_vehicle_center[2] -= 1.22
    return P_vehicle_center

def project_image_to_vehicle_coordinate_system(trajectory_image: np.ndarray, intrinsic_matrix: np.ndarray, dist_move_forward: np.ndarray):
    """画像座標系の軌跡を車両中心座標系に逆変換する関数"""
    road_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])

    # 画像座標系から車両中心座標系に逆変換
    inverse_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)
    trajectory_vehicle = []
    for P_image, move_x in zip(trajectory_image, dist_move_forward):
        P_camera = image_to_camera(P_image, inverse_intrinsic_matrix, move_x)
        # if move_x > 0:
        #     P_camera = image_to_camera(P_image, inverse_intrinsic_matrix, move_x)
        # else:
        #     P_image[1] = -P_image[1]
        #     P_camera = image_to_camera(P_image, inverse_intrinsic_matrix, - move_x)
        # # # Z座標が正である（カメラから前方にある）点のみ処理
        # # if P_camera[2] > 0:
        P_vehicle = camera_to_vehicle_center(P_camera, road_to_camera)
        trajectory_vehicle.append(P_vehicle)
    return np.array(trajectory_vehicle)


def get_trajectory(row):
    TARGET_COLUMNS = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']

    # データフレームのターゲット情報から可視化可能なtrajectoryに変換
    pivot_df = row[TARGET_COLUMNS].to_frame().reset_index()

    pivot_df.columns = ['coordinate', 'value']

    # 座標軸(x,y,z)と番号(0-5)を正規表現で抽出
    # 例：'x_0' -> axis='x', number='0'
    pivot_df[['axis', 'number']] = pivot_df['coordinate'].str.extract(r'([xyz])_(\d+)')

    # ピボットテーブルを作成：
    # - インデックス：番号(0-5)
    # - カラム：座標軸(x,y,z)
    # - 値：対応する座標値
    trajectory = pivot_df.pivot(index='number', columns='axis', values='value')

    # インデックスを数値型に変換
    trajectory.index = trajectory.index.astype(int)

    # インデックスでソートし、numpy配列に変換
    trajectory = trajectory.sort_index().values
    return trajectory

# CosineAnnealingWarmupが最後min_lrの収束するように書き換え
def _get_cosine_schedule_with_warmup_min_lr(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr: float, initial_lr: float
):
    if current_step < num_warmup_steps:
        # Warmup phase: linearly increase from 0 to initial_lr
        return float(current_step) / float(max(1, num_warmup_steps))
    
    # Cosine phase
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_lr = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    
    # Scale cosine_lr to range [min_lr / initial_lr, 1.0]
    min_lr_ratio = min_lr / initial_lr
    return max(min_lr_ratio, cosine_lr * (1.0 - min_lr_ratio) + min_lr_ratio)

def get_cosine_schedule_with_warmup_min_lr(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr: float = 1e-7, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to `min_lr`, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        min_lr (`float`):
            The minimum learning rate to which the cosine schedule will decay.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the default is to just decrease from the max value to `min_lr`
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Get the initial learning rate from the optimizer
    initial_lr = optimizer.defaults['lr']

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_min_lr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr=min_lr,
        initial_lr=initial_lr,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

###################
### Dataset
###################
class VehicleTrajectoryDataset(Dataset):
    def __init__(self, df, trajectories, config, mode="train", transform=None):
        self.df = df
        self.trajectories = trajectories
        self.size = config.size
        self.image_dir = config.image_dir
        self.transform = transform
        self.mode = mode
        self.feat_cols = ["vEgo", "aEgo", "steeringAngleDeg", "steeringTorque", "brakePressed", "gas", "gasPressed", "leftBlinker", "rightBlinker"]
    
    def __len__(self):
        return len(self.df)
    
    def load_images(self, id):
        suffixes = ["", "-0.5", "-1.0"]
        images = []
        for suffix in suffixes:
            image_path = os.path.join(self.image_dir, f"{id.split('_')[0]}", f"{id.split('_')[1]}", f"image_t{suffix}.png")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images
    
    def __getitem__(self, idx):
        id = self.df.iloc[idx].ID
        features = torch.tensor(self.df.iloc[idx][self.feat_cols].values.astype("float")).float()

        image1, image2, image3 = self.load_images(id)
        if self.transform:
            if self.mode.lower() != "test":
                augmented = self.transform(image=image1)
                image1 = augmented["image"] / 255.
                # Assume same transformation for all images using ReplayCompose
                replay_params = augmented['replay']
                image2 = A.ReplayCompose.replay(replay_params, image=image2)["image"] / 255.
                image3 = A.ReplayCompose.replay(replay_params, image=image3)["image"] / 255.
            else:
                image1 = self.transform(image=image1)["image"] / 255.
                image2 = self.transform(image=image2)["image"] / 255.
                image3 = self.transform(image=image3)["image"] / 255.

        # チャネル軸方向に結合
        images = torch.cat([image1, image2, image3], dim=0)

        if self.mode.lower() != "test":
            trajectory = self.trajectories[idx].copy()
            return images.float(), features, torch.tensor(trajectory).float()
        else:
            return images.float(), features
    


###################
### Model
###################
class VehicleTrajectoryModel(pl.LightningModule):
    def __init__(self, backbone, learning_rate, weight_decay, num_training_steps, num_warmup_steps, config,
                 pretrained=False, drop_rate=0., drop_path_rate=0., drop_rate_last=0.):
        super().__init__()

        hp = vars(config)
        hp.update({"pretrained": pretrained, "drop_rate": drop_rate, "drop_path_rate": drop_path_rate, "drop_rate_last": drop_rate_last,
                   "num_training_steps": num_training_steps, "num_warmup_steps": num_warmup_steps})
        self.save_hyperparameters(hp)

        self.model = timm.create_model(
            backbone,
            in_chans=9,
            num_classes=1,
            features_only=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )

        if 'efficient' in backbone:
            self.hdim = self.model.conv_head.out_channels
            self.model.classifier = nn.Identity()
        elif any([prefix in backbone for prefix in ['convnext', 'swin']]):
            self.hdim = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
        elif 'resnet' in backbone:
            self.hdim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif "dla34" in backbone:
            self.hdim = 512
        else:
            raise ValueError(f"Invalid backbone: {backbone}")
        

        self.feature_layer = nn.Linear(9, 128)

        self.head_traj = nn.Sequential(
            nn.Linear(self.hdim + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6*3)
        )

        self.loss_fn = nn.L1Loss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
    
    def forward(self, images, features):
        x = self.model(images) # (B, hdim)

        features = self.feature_layer(features) # (B, 128)
        x = torch.cat([x, features], dim=-1) # (B, hdim + 128)

        pred_trajectory = self.head_traj(x).reshape(-1, 6, 3) # (B, 6, 3)
        return pred_trajectory
    
    def training_step(self, batch, batch_idx):
        images, features, trajectory_camera = batch
        pred_trajectory = self(images, features)
        loss = self.loss_fn(pred_trajectory, trajectory_camera)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        # Log the learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, features, trajectory_camera = batch
        pred_trajectory = self(images, features)

        loss = self.loss_fn(pred_trajectory, trajectory_camera)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup_min_lr(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]



if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Lightning Version: {pl.__version__}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Albumentations Version: {A.__version__}")
    print(f"TIMM Version: {timm.__version__}")
    
    ###################
    ### Set Configurations
    ###################
    config = dict(
        exp = "xx",
        input_dir = "xxxx",
        output_dir = "xxxx",
        image_dir = "xxxx",
        seed = 0,
        n_folds = 5,
        model_name = "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        batch_size = 16,
        num_epochs = 30,
        learning_rate = 5e-4,
        weight_decay = 1e-6,
        accumulate_grad_batches = 1,
        size = 224,
        gradient_clip_val = None,
        num_workers = 8,
        drop_rate = 0.,
        drop_path_rate = 0.,
        drop_rate_last = 0.,
        precision = "bf16-mixed",
        stage = "first"
    )

    # もし追加で引数を追加したければ柔軟に認識
    parser = argparse.ArgumentParser()
    _, given_args = parser.parse_known_args()
    given_args_dict = {}
    for arg in given_args:
        if arg.startswith('--'):
            key = arg.lstrip('--')
            value = True
            if len(given_args) > given_args.index(arg) + 1:
                next_arg = given_args[given_args.index(arg) + 1]
                if not next_arg.startswith('--'):
                    value = convert_string(next_arg)
            given_args_dict[key] = value
    # Update
    if given_args_dict:
        print("Given Arguments: ", given_args_dict)
    config = config | given_args_dict
    config = Namespace(**config)


    ## Define augumentation settings
    transform = {
        "train": A.ReplayCompose([
            A.Resize(height=config.size, width=config.size, interpolation=cv2.INTER_CUBIC),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2()
            ]),
        "valid": A.ReplayCompose([
            A.Resize(height=config.size, width=config.size, interpolation=cv2.INTER_CUBIC),
            ToTensorV2()
            ])
        
    }
    
    ###################
    ### Load Data
    ###################
    train_df = pd.read_csv(os.path.join(config.output_dir, "train_features_w_cluster.csv"))
    train_ids = train_df["ID"].unique()
    test_df = pd.read_csv(os.path.join(config.output_dir, "test_features_w_cluster.csv"))
    test_ids = test_df["ID"].unique()

    feat_cols = ["vEgo", "aEgo", "steeringAngleDeg", "steeringTorque", "brakePressed", "gas", "gasPressed", "leftBlinker", "rightBlinker"]
    all_feat = pd.concat([train_df[feat_cols], test_df[feat_cols]], axis=0).reset_index(drop=True)
    all_feat.vEgo = all_feat.vEgo / 30
    all_feat.aEgo = all_feat.aEgo
    all_feat.steeringAngleDeg = all_feat.steeringAngleDeg / 400
    all_feat.steeringTorque = all_feat.steeringTorque / 600
    for feat in feat_cols:
        if all_feat[feat].dtype == "float":
            all_feat[feat] = StandardScaler().fit_transform(all_feat[feat].values.reshape(-1, 1)).flatten()
    train_df[feat_cols] = all_feat[:len(train_df)].values
    test_df[feat_cols] = all_feat[len(train_df):].values

    # CV splitting
    scene_ids = train_df["ID"].apply(lambda x: x.split("_")[0])
    # skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    train_df["fold"] = -1
    for fold, (train_idx, valid_idx) in enumerate(sgkf.split(train_df, train_df["cluster"], groups=scene_ids)):
        train_df.loc[valid_idx, "fold"] = fold
    
    # Fetch trajectories in advance
    train_trajectories = np.stack(Parallel(n_jobs=4)(delayed(get_trajectory)(row) for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Fetching trajectories....")), axis=0)
    train_trajectories = train_trajectories.astype("float")
    
    ###################
    ### Run
    ###################
    print("Start training....")
    save_path = os.path.join(config.output_dir, f"{config.exp}_{config.model_name.split('.')[0]}")
    os.makedirs(save_path, exist_ok=True)

    for fold in range(config.n_folds):
        print("="*20)
        print("="*6, f"Fold {fold}", "="*6)
        print("="*20)

        trn_idxes = np.where(train_df["fold"] != fold)[0]
        val_idxes = np.where(train_df["fold"] == fold)[0]

        trn_ds = VehicleTrajectoryDataset(train_df.iloc[trn_idxes].reset_index(drop=True), train_trajectories[trn_idxes], config, mode="train", transform=transform["train"])
        val_ds = VehicleTrajectoryDataset(train_df.iloc[val_idxes].reset_index(drop=True), train_trajectories[val_idxes], config, mode="valid", transform=transform["valid"])
        
        if config.stage == "first":
            ## First, train the model with only the head
            print("--- First Training ---")
            num_training_steps = (train_df[train_df["fold"] != fold].shape[0] // config.batch_size) * (config.num_epochs - 1)
            num_warmup_steps = train_df[train_df["fold"] != fold].shape[0] // config.batch_size
            trn_dl = DataLoader(trn_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
            val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False)

            # Model
            pl.seed_everything(config.seed)
            model = VehicleTrajectoryModel(config.model_name, config.learning_rate, config.weight_decay, num_training_steps, num_warmup_steps, config,
                                        pretrained=True, drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate, drop_rate_last=config.drop_rate_last)
            for param in model.model.parameters():
                param.requires_grad = False
            
            # ModelCheckpoint callback
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=save_path,
                filename=f'best-checkpoint-fold{fold}-seed{config.seed}-first',
                save_top_k=1,
                mode='min'
            )

            logger = CSVLogger(save_dir=save_path,
                            name="logs")
            
            # Trainer with mixed precision and ModelCheckpoint
            trainer = pl.Trainer(
                max_epochs=config.num_epochs,
                precision=config.precision,  # Enable mixed precision training
                callbacks=[checkpoint_callback],
                check_val_every_n_epoch=1,
                logger=logger,
                accumulate_grad_batches=config.accumulate_grad_batches,
                gradient_clip_val = config.gradient_clip_val,
                log_every_n_steps=10
            )

            # Train the model
            trainer.fit(model, trn_dl, val_dl)

            del model, trainer, trn_ds, val_ds, trn_dl, val_dl
            gc.collect()
            torch.cuda.empty_cache()

        elif config.stage == "second":
            # Second, train the whole model
            print("--- Second Training ---")
            # config.batch_size = 128
            # config.learning_rate = 5e-4
            num_training_steps = (train_df[train_df["fold"] != fold].shape[0] // config.batch_size) * (config.num_epochs - 1)
            num_warmup_steps = train_df[train_df["fold"] != fold].shape[0] // config.batch_size
            trn_dl = DataLoader(trn_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
            val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False)

            # Model
            pl.seed_everything(config.seed)
            model = VehicleTrajectoryModel(config.model_name, config.learning_rate, config.weight_decay, num_training_steps, num_warmup_steps, config,
                                        pretrained=True, drop_rate=config.drop_rate, drop_path_rate=config.drop_path_rate, drop_rate_last=config.drop_rate_last)
            
            ## stage1のモデルをロード
            # backboneの重みも更新可能に変更
            state_dict = torch.load(os.path.join(save_path, f'best-checkpoint-fold{fold}-seed{config.seed}-first.ckpt'), map_location="cpu")["state_dict"]
            model.load_state_dict(state_dict)            
            for param in model.model.parameters():
                param.requires_grad = True

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=save_path,
                filename=f'best-checkpoint-fold{fold}-seed{config.seed}-second',
                save_top_k=1,
                mode='min'
            )
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=6, verbose=False, mode="min")

            logger = CSVLogger(save_dir=save_path,
                            name="logs")
            
            # Trainer with mixed precision and ModelCheckpoint
            trainer = pl.Trainer(
                max_epochs=config.num_epochs,
                precision=config.precision,  # Enable mixed precision training
                callbacks=[checkpoint_callback, early_stop_callback],
                check_val_every_n_epoch=1,
                logger=logger,
                accumulate_grad_batches=config.accumulate_grad_batches,
                gradient_clip_val = config.gradient_clip_val,
                log_every_n_steps=10
            )

            trainer.fit(model, trn_dl, val_dl)

            del model, trainer, trn_ds, val_ds, trn_dl, val_dl
            gc.collect()
            torch.cuda.empty_cache()
