#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
cd ../

echo "************************************"
echo "* Running Forward Looking Modeling *"
echo "************************************"

echo "*****************************************"
echo "* Training team_batting & pitcher model *"
echo "*****************************************"
python3 run_modeling.py --type "team_batting" "pitcher" \
             --n_games_context 10 --context_max_len 1124 \
             --model_type "vanilla" --attn_mask_type "bidirectional" \
             --completion_max_len 116 \
             --n_warmup_iters 4000 --batch_size 24 \
             --lr 5e-4 --l2 1e-4 --epochs 35 \
             --completion_max_len 156 \
             --use_player_id F --v2_encoder T \
             --dataset_size_train 100000 \
             --dataset_size_else 12500 \
             --n_data_workers 5 \
             --n_data_workers_else 4 \
             --ben_vocab T \
             --use_explicit_test_masked_indices F \
             --reduced_event_map T \
             --tie_weights T \
             --xent_label_smoothing 0.0 \
             --ordinal_pos_embeddings T \
             --player_cls T \
             --use_ball_data T \
             --mask_ball_data F \
             --drop_ball_data F \
             --predict_ball_data F \
             --starting_pitcher_only F \
             --gsd_embd_dim 384 \
             --context_only T \
             --v2_player_attn F \
             --do_group_gathers T \
             --norm_first T