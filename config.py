import torch
import os

exp_name = "mobileNetV3-Large_CELoss"
test_result_dir = "/home/jykim/work/MobileNetV3-Large/csvs & confusion matrix"
checkpoint_path = "/home/jykim/work/MobileNetV3-Large/result/lightning_logs/large_ce/checkpoints/epoch=19-val_loss=0.0000-val_acc=1.0000-v1.ckpt"
confusion_labels = ['Vehicle', 'Footsteps', 'Other']

num_gpu = [0]
os.environ["CUDA_VISIBLE_DEVICES"] =f"{num_gpu[0]}"
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
max_epoch = 20

random_seed = 42

classes_num = 3
batch_size = 256
num_workers = 8
####################
dataset_path = "/home/jykim/work/HTS-AT/aihub_dataset/v2_MotorAndCar"
save_path = "/home/jykim/work/MobileNetV3-Large/result"
####################

# for signal processing
sample_rate = 16000 # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * 10 # audio_set 10-sec clip
window_size = 1024
hop_size = 160 # 160 for scv2, 320 for audioset and esc-50
mel_bins = 64
fmin = 50
fmax = 14000
####################
window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None