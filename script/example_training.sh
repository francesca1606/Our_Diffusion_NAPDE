

accelerate launch train/train.py \
        --save \
        --wandb \
        --lr=3e-4 \
        --conditional \
        --nb_epochs=200 \
        --batch_size=16 \
        --diffusion_mode="classic" \
        --dataset="6000_data" \
