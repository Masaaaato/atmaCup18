"""
## bacboneの特徴量に走行時特徴量をconcatした後, 同一sceneについてLSTMで時系列処理を行うモデル
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
        self.scene_ids = df["scene_id"].unique()
        self.trajectories = trajectories
        self.size = config.size
        self.image_dir = config.image_dir
        self.transform = transform
        self.mode = mode
        self.feat_cols = ["vEgo", "aEgo", "steeringAngleDeg", "steeringTorque", "brakePressed", "gas", "gasPressed", "leftBlinker", "rightBlinker"]

    def __len__(self):
        return len(self.scene_ids)
    
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
        scene_id = self.scene_ids[idx]
        idxes = np.where(self.df["scene_id"] == scene_id)[0]

        stacked_images = []
        stacked_features = []
        stacked_trajectories = []

        for i in idxes:
            id = self.df.iloc[i].ID
            features = torch.tensor(self.df.iloc[i][self.feat_cols].values.astype("float")).float()

            trajectory = torch.tensor(self.trajectories[i]).float() if self.trajectories else None

            # Laod images and apply augmentations while taking trajectory as keypoints
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
            
            # Normalize
            # image1 = (image1 - self.mean) / self.std
            # image2 = (image2 - self.mean) / self.std
            # image3 = (image3 - self.mean) / self.std

            # チャネル軸方向に結合
            images = torch.cat([image1, image2, image3], dim=0)

            stacked_images.append(images)
            stacked_features.append(features)
            stacked_trajectories.append(trajectory)

        stacked_images = torch.stack(stacked_images, dim=0)
        stacked_features = torch.stack(stacked_features, dim=0)
        stacked_trajectories = torch.stack(stacked_trajectories, dim=0) if self.trajectories else None

        if self.mode.lower() != "test":
            return stacked_images, stacked_features, stacked_trajectories
        else:
            return scene_id, stacked_images, stacked_features

def collate_fn(batch):
    # get sequence lengths
    seq_lengths = torch.tensor([x[0].size(0) for x in batch])

    images = torch.cat([x[0] for x in batch], dim=0)
    features = torch.cat([x[1] for x in batch], dim=0)
    # trajectories = torch.cat([x[2] for x in batch], dim=0) # 推論時は不要

    # return images, features, trajectories, seq_lengths
    return images, features, seq_lengths

###################
### Model
###################
# 推論時はnn.Moduleを継承したクラスを作成
# pl.LightningModuleをnn.Moduleに換えて余分な引数を削除
class VehicleTrajectoryModel(nn.Module):
    def __init__(self, backbone, pretrained=False, drop_rate=0., drop_path_rate=0., drop_rate_last=0.):
        super().__init__()

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

        self.lstm = nn.LSTM(input_size=self.hdim + 128, hidden_size=128, num_layers=config.num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(128*2, 6*3)
    
    def forward(self, images, features, seq_lengths):
        x = self.model(images) # (B, hdim)

        features = self.feature_layer(features) # (B, 128)
        x = torch.cat([x, features], dim=-1) # (B, hdim + 128)
        
        lens = seq_lengths.cpu().tolist()
        sequences = torch.split(x, lens)
        padded_sequences = pad_sequence(sequences, batch_first=True)

        # ソート
        sorted_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        padded_sequences = padded_sequences[perm_idx]

        # パック
        packed_sequences = pack_padded_sequence(padded_sequences, sorted_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM
        output, _ = self.lstm(packed_sequences)

        # 出力を元に戻す
        unpadded_output, _ = pad_packed_sequence(output, batch_first=True)

        # ソート時のインデックスを逆順に変換
        _, original_idx = perm_idx.sort(0)

        # 元の順番に戻す
        unpadded_output_original_order = []
        for idx, len_ in zip(original_idx, lens):
            unpadded_output_original_order.append(unpadded_output[idx, :len_])
        unpadded_output_original_order = torch.cat(unpadded_output_original_order, dim=0)

        pred_trajectory = self.head(unpadded_output_original_order).reshape(-1, 6, 3) # (B, 6, 3)
        return pred_trajectory



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
        num_layers = 2,
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
    train_ids = train_df["ID"].values
    train_df["scene_id"] = train_df["ID"].apply(lambda x: x.split("_")[0])
    train_df["frame_id"] = train_df["ID"].apply(lambda x: x.split("_")[1]).astype("int")
    train_df = train_df.sort_values(["scene_id", "frame_id"], ascending=True).reset_index(drop=True)
    
    test_df = pd.read_csv(os.path.join(config.output_dir, "test_features_w_cluster.csv"))
    test_ids = test_df["ID"].values
    test_df["scene_id"] = test_df["ID"].apply(lambda x: x.split("_")[0])
    test_df["frame_id"] = test_df["ID"].apply(lambda x: x.split("_")[1]).astype("int")
    test_df = test_df.sort_values(["scene_id", "frame_id"], ascending=True).reset_index(drop=True)

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
    # 適宜変更してください.
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
    ### Inference
    ###################
    print("Start inference....")
    save_path = os.path.join(config.output_dir, f"{config.exp}_{config.model_name.split('.')[0]}")
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初期化
    test_preds = np.zeros((len(test_df), 6, 3))
    oof_preds = np.zeros((len(train_df), 6, 3))

    for fold in range(config.n_folds):
        print("="*20)
        print("="*6, f"Fold {fold}", "="*6)
        print("="*20)

        val_idxes = np.where(train_df["fold"] == fold)[0]

        val_ds = VehicleTrajectoryDataset(train_df.iloc[val_idxes].reset_index(drop=True), None, config, mode="test", transform=transform["valid"])
        test_ds = VehicleTrajectoryDataset(test_df, None, config, mode="test", transform=transform["valid"])

        val_dl = DataLoader(val_ds, batch_size=config.batch_size,
                                collate_fn=collate_fn, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True, drop_last=False)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size,
                                collate_fn=collate_fn, shuffle=False,
                                num_workers=config.num_workers, pin_memory=True, drop_last=False)

        # Model
        model = VehicleTrajectoryModel(config.model_name)
        model.load_state_dict(torch.load(os.path.join(save_path, f'best-checkpoint-fold{fold}-seed{config.seed}-second.ckpt'), map_location="cpu")["state_dict"])
        model.eval().to(device)

        # Inference
        val_preds = []
        for images, features, seq_lengths in tqdm(val_dl, desc="Validation Inference...."):
            images, features, seq_lengths = images.to(device), features.to(device), seq_lengths.to(device)
            with torch.no_grad():
                pred = model(images, features, seq_lengths)
                val_preds.append(pred.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        oof_preds[val_idxes] = val_preds

        test_preds_fold = []
        for images, features, seq_lengths in tqdm(test_dl, desc="Test Inference...."):
            images, features, seq_lengths = images.to(device), features.to(device), seq_lengths.to(device)
            with torch.no_grad():
                pred = model(images, features, seq_lengths)
                test_preds_fold.append(pred.cpu().numpy())
        test_preds_fold = np.concatenate(test_preds_fold, axis=0)
        test_preds += test_preds_fold / config.n_folds
        
    oof_mae =np.mean(np.abs(train_trajectories - oof_preds))
    print(f"OOF MAE: {oof_mae:.6f}")
    
    # Save OOF predictions
    targets = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4', 'x_5', 'y_5', 'z_5']
    oof_df = pd.DataFrame(oof_preds.reshape(-1, 18), columns=targets, index=train_df["ID"]).loc[train_ids].reset_index(names="ID")
    oof_df.to_csv(os.path.join(save_path, f"oof_preds.csv"), index=False)

    # Save test predictions
    sub_df = pd.DataFrame(test_preds.reshape(-1, 18), columns=targets, index=test_df["ID"]).loc[test_ids].reset_index(drop=True)
    sub_df.to_csv(os.path.join(save_path, f"submission.csv"), index=False)
