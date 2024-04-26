from datasets import load_dataset

dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0', download_mode="force_redownload")
dataset.save_to_disk("/mnt/swordfish-pool2/horvitz/cnn_dailymail")
