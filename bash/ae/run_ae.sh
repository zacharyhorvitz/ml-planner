# python text_autoencoder/autoencoder/train.py \
# --log_interval 10 --val_interval 10 --batch_size 2 --valid_size 1024 \
# --epoch 1 \
# --num_feature 16 --sentence_len 256 \
# --train_pt_dir data-bin/dummy_data/parsed/train --dev_pt_dir data-bin/dummy_data/parsed/dev \
# --h_noiser vae --h_noiser_ratio 0.00001 \
# --h_tanh \
# --enc_model bert-large-uncased \
# --dec_model gpt2-medium \
# --load_dec True \
# --latent_size 1024 \
# --share_gpts True \

# Probably need to fix bucket dataloader issue before doing a distributed train

python text_autoencoder/autoencoder/train.py \
--log_interval 10 --val_interval 500 --batch_size 12 --valid_size 1024 \
--save_interval 1000 \
--epoch 20 \
--num_feature 16 --sentence_len 256 \
--train_pt_dir /mnt/swordfish-pool2/horvitz/cnn_dailymail_sum_tokenized/ --dev_pt_dir data-bin/dummy_data/parsed/dev \
--h_noiser vae --h_noiser_ratio 0.00001 \
--h_tanh \
--gpus 1 \
--world_size 1 \
--lr 0.000125 \
--enc_model bert-large-uncased \
--dec_model gpt2-medium \
--load_dec True \
--latent_size 1024 \
--share_gpts True \
--save_dir /mnt/swordfish-pool2/horvitz/ae_models/ \

# python text_autoencoder/autoencoder/train.py \
# --log_interval 10 --val_interval 500 --batch_size 12 --valid_size 1024 \
# --save_interval 1000 \
# --epoch 20 \
# --num_feature 16 --sentence_len 256 \
# --train_pt_dir /mnt/swordfish-pool2/horvitz/cnn_dailymail_sum_tokenized/ --dev_pt_dir data-bin/dummy_data/parsed/dev \
# --h_noiser vae --h_noiser_ratio 0.00001 \
# --h_tanh \
# --gpus 1 \
# --world_size 1 \
# --lr 0.0000625 \
# --enc_model bert-large-uncased \
# --dec_model gpt2-medium \
# --load_dec True \
# --latent_size 1024 \
# --share_gpts True \
# --save_dir /mnt/swordfish-pool2/horvitz/ae_models/ \
