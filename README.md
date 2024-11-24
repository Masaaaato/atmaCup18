# atmaCup18
これは[#18 Turing × atmaCup](https://www.guruguru.science/competitions/25)に関連するコードレポジトリです.  
最後のblendingで発生させたバグのせいで間に合わず，直後のlate subでpublic 0.1965, private 0.1834 (10位相当)です.  

## Overview
1. NN1
   CNN backboneに走行時特徴量を結合してMLP  
   <img src="https://github.com/user-attachments/assets/2f2b4364-e537-4d8e-9018-050264f22d85" alt="NN1" width="300">

3. NN2
   CNN backboneに走行時特徴量を結合した後，同一sceneについてLSTM  
   <img src="https://github.com/user-attachments/assets/889ca448-caf4-4720-a7d8-2a0773d7a5f3" alt="NN1" width="300">

5. image embedding ([RSUD20K-DETR](https://github.com/hasibzunair/RSUD20K)のbackbone出力)  
   車載カメラ画像で学習済みのDETRモデルのbacbone CNN出力を取得. 3frames間の類似度も計算
6. GBDT  
   NN1/NN2のOOF, image embeddingのSVDとframe間相関, 走行時特徴量のscene間統計量やshiftやdiff.
7. blending  
   同一scene枚数に応じて、1-2, 3-6枚を分けてそれぞれでNelder-mead

 ## Examples
 1. NN1
    ```bash
    # Training
    python -u run/cnn_w_driving_features --batch_size 128 --learning_rate 2e-3 --weight_decay 1e-6 --num_epochs 20 --stage "first"
    python -u run/cnn_w_driving_features --batch_size 128 --learning_rate 5e-4 --weight_decay 1e-6 --num_epochs 30 --stage "second"

    # Infenence and OOF
    python -u run/cnn_w_driving_features_inference    
    ```

2. NN2
   ```bash
   # Training
    python -u run/cnn_lstm_w_driving_features --batch_size 128 --learning_rate 2e-3 --weight_decay 1e-6 --num_epochs 20 --num_layers 2 --stage "first"
    python -u run/cnn_lstm_w_driving_features --batch_size 16 --learning_rate 2e-4 --weight_decay 1e-6 --num_epochs 30 --num_layers 2 --stage "second"

    # Infenence and OOF
    python -u run/cnn_lstm_w_driving_features_inference  
   ```

3. image embedding  
[`RSUD20K_DETR_embs.ipynb`](https://github.com/Masaaaato/atmaCup18/blob/main/run/RSUD20K_DETR_embs.ipynb)
を参照して下さい

5. GBDT  
[`XGB_convnext_oof.ipynb`](https://github.com/Masaaaato/atmaCup18/blob/main/run/XGB_convnext_oof.ipynb)
を参照して下さい

7. blending
```python
import pandas as pd
from scipy.optimize import minimize

# 同一scene枚数を取得
train_df = pd.read_csv(os.path.join(config.output_dir, "train_features_w_cluster.csv"))
train_num_scenes = train_df["scene_id"].map(train_scene_id2num).values 
test_df = pd.read_csv(os.path.join(config.output_dir, "test_features_w_cluster.csv"))
test_num_scenes = test_df["scene_id"].map(test_scene_id2num).values

# target trajectories
train_targets = train_df[targets].values

# oofが4つ, testが4つあると仮定します
oof_blended = np.zeros_like(oof1.values)
test_blended = np.zeros_like(test1.values)
# 同一scene数が1~2枚, 3~6枚で分けてNelder-Mead
for num_scenes in [[1, 2], [3, 4, 5, 6]]:
    ind = np.where(np.isin(train_num_scenes, num_scenes))[0]
    tmp_train_targets = train_targets[ind].copy()
    predictions = [oof1.iloc[ind].values,
                    oof2.iloc[ind].values,
                    oof3.iloc[ind].values,
                    oof4.iloc[ind].values]
    # initialize
    initial_weights = np.ones(len(predictions)) / len(predictions)

    def objective(weights):
        weights = np.clip(weights, 0, None)
        weights /= weights.sum()

        blended_predictions = sum(w * p for w, p in zip(weights, predictions))
        score = np.mean(np.abs(blended_predictions - tmp_train_targets))
        return score

    result = minimize(
        objective,
        initial_weights,
        method="Nelder-Mead",
        options={"maxiter": 1000, "disp": True}
    )

    optimal_weights = np.clip(result.x, 0, None)
    optimal_weights /= optimal_weights.sum()
    optimal_value = result.fun

    print("Optimal weights:", optimal_weights)
    print("Optimal value:", optimal_value)

    tmp_oof = sum(w * p for w, p in zip(optimal_weights, predictions))
    oof_blended[ind] = tmp_oof

    # Blending for test
    test_ind = np.where(np.isin(test_num_scenes, num_scenes))[0]

    test_predictions = [test1.iloc[test_ind].values,
                        test2.iloc[test_ind].values,
                        test3.iloc[test_ind].values,
                        test4.iloc[test_ind].values]

    test_preds = sum(w * p for w, p in zip(optimal_weights, test_predictions))  
    test_blended[test_ind] = test_preds
```
   
