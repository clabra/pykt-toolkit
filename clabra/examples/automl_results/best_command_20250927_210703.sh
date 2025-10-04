
# 🏆 REPRODUCE BEST RESULT (AUC: 0.7245)
cd /workspaces/pykt-toolkit/examples
python wandb_gainakt2_train.py \
    --dataset_name=assist2015 \
    --use_wandb=0 \
    --d_model=192 \
    --learning_rate=1.23e-04 \
    --dropout=0.217 \
    --num_encoder_blocks=6 \
    --d_ff=640 \
    --n_heads=16 \
    --num_epochs=10 \
    --seed=42 \
    --fold=0
