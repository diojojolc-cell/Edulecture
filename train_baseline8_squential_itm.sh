
export CUDA_VISIBLE_DEVICES=0

python train_baseline8_dataparallel_squential_itm.py \
  --arch baseline_8_squential_itm \
  --exp_name clip \
  --batch_size 64 \
  --noclip_lr 3e-5 \
  --transformer_dropout 0.3 \
  --dataset_name baseline \
  --stochasic_trials 20 \
  --gpu 0 \
  --num_epochs 50 \
  --support_loss_weight 0.8 \
  --embed_dim 768 \
  --num_frames 12 \
  --train_data_path 
  --test_data_path