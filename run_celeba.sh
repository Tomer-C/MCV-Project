
#!/bin/bash
torchrun --nproc_per_node=8 --master_port 1236 main.py \
        --run_name 'UNet' \
        --ddp \
        --project_name 'IGN CelebA' \
        --batch_size 256 \
        --dataset_name 'CelebA' \
        --model_name 'MUNet' \
        --lr 0.0001 \
        --beta1 0.5 \
        --beta2 0.999 \
        --discriminative_loss_factor 10. \
        --in_manifold_loss_factor 1.0 \
        --out_of_manifold_loss_factor 1.0 \
        --discriminative_loss_clamp_ratio 1. \
        --epoch_begin_idempotence_losses 0 \
        --epoch_begin_gen_input 9999 \
        --noise_type 'normal' \
        --loss_p 1.0 \
        --match_means_b_factor 0. \
        --match_stds_b_factor 0. \
        --discriminative_loss_clamp_val 9999. \
        --num_epochs 10000 \
        # --use_consine_scheduler \
        # --resume '/home/assafsho/IGN/IGN CelebA_checkpoint/ckpt_416.pt'
