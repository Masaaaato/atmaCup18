# atmaCup18
これは[#18 Turing × atmaCup](https://www.guruguru.science/competitions/25)に関連するコードレポジトリです.  
菜後のblendingで発生させたバグのせいで間に合わず，直後のlate subでpublic 0.1965, private 0.1834 (10位相当)です.  

## Overview
1. NN1
   CNN backboneに走行時特徴量を結合してMLP
   <img width="385" alt="image" src="https://github.com/user-attachments/assets/7074b769-8e59-4d66-9ee7-103052c44e20">
2. NN2
   CNN backboneに走行時特徴量を結合した後，同一sceneについてLSTM
   <img width="366" alt="image" src="https://github.com/user-attachments/assets/d77ce6e6-c9a4-4e3b-a026-0f588c678547">
3. image embedding ([RSUD20K-DETR](https://github.com/hasibzunair/RSUD20K)のbackbone出力)
   車載カメラ画像で学習済みのDETRモデルのbacbone CNN出力を種得. 3frames間の類似度も計算
4. GBDT
   NN1/NN2のOOF, image embeddingのSVDとframe間相関, 走行時特徴量のscene間統計量やshiftやdiff.
5. blending
   同一scene枚数に応じて、1~2, 3~6枚を分けてそれぞれでNelder-mead

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

3. image embeding
   
   
