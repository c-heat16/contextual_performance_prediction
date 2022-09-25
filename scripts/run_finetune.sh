#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export N_WORKERS=4
export N_WORKERS_ELSE=5

cd ../

echo "*************************"
echo "* Running FL Finetuning *"
echo "*************************"

python3 finetune_fl_models.py --model_types "vanilla" \
  --batter_ckpt "/home/czh/nvme1/SportsAnalytics/out/fl_modeling/20220922-105649/models/model_34e.pt" \
  --pitcher_ckpt "/home/czh/nvme1/SportsAnalytics/out/fl_modeling/20220918-181922/models/model_34e.pt" \
  --model_ckpt "none" \
  --single_model F --entity_models T \
  --batch_size 6 --n_warmup_iters 200 --epochs 6 \
  --n_data_workers $N_WORKERS --n_data_workers_else $N_WORKERS_ELSE \
  --l2 1e-6 --lr 2e-4 \
  --single_pitcher_batter_completion T \
  --pitcher_targets k h --pitcher_scalars 17 17 \
  --batter_targets k h --batter_scalars 6 7 \
  --binary_hit_preds F --cyclic_lr F --cosine_lr T \
  --use_matchup_data F --do_ln F \
  --loss_type "mse" --batter_weight 1 \
  --pitcher_weight 1 --batter_has_hit_weight 1 \
  --n_grad_accum 2