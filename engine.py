import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score, confusion_matrix
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import numpy as np 
import config

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import pdb

device = config.device

spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, win_length=config.window_size, window=config.window, center=config.center, pad_mode=config.pad_mode, freeze_parameters=True).to(device)
logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax, ref=config.ref, amin=config.amin, top_db=config.top_db, freeze_parameters=True).to(device)

class Engine(pl.LightningModule) :
    def __init__(self, model) :
        super().__init__()
        self.model = model
    
    def val_evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"val_acc": acc} 
    
    def test_evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"test_acc": acc} 
    
    def forward(self, x) :
        return self.model(x)
    
    def training_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat = self.model(mel_spec_x)
        # pdb.set_trace()
        
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat = self.model(mel_spec_x)
        
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=config.batch_size)
        
        return [y_hat.detach(), y.detach()]
    
    def validation_epoch_end(self, validation_step_outputs) :
        y_hat = torch.cat([x[0] for x in validation_step_outputs], dim=0)
        y = torch.cat([x[1] for x in validation_step_outputs], dim=0)
        
        gather_pred = y_hat.cpu().numpy()
        gather_target = y.cpu().numpy()
        gather_target = np.argmax(gather_target, 1)
        
        metric_dict = self.val_evaluate_metric(gather_pred, gather_target)
        self.log("val_acc", metric_dict["val_acc"], on_epoch=True, prog_bar=True, logger=True, batch_size=config.batch_size)
        
    def test_step(self, batch, batch_idx) :
        x, y = batch["waveform"].to(device), batch["target"].to(device)
        spec_x = spectrogram_extractor(x)
        mel_spec_x = logmel_extractor(spec_x)
        mel_spec_x = mel_spec_x.repeat(1, 3, 1, 1)
        y_hat = self.model(mel_spec_x)
        
        return [y_hat.detach(), y, batch['audio_name']]
    
    def test_epoch_end(self, test_step_outputs):
        pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
        audio_name = [name for sublist in [d[2] for d in test_step_outputs] for name in sublist]
        
        gather_pred = pred.cpu().numpy()
        gather_target = target.cpu().numpy()
        gather_target = np.argmax(gather_target, 1)
        
        # pdb.set_trace()
        
        metric_dict = self.test_evaluate_metric(gather_pred, gather_target)
        
        # ================== save csv ====================
        
        df = pd.DataFrame()
        df['file'] = audio_name
        df['model_output'] = gather_pred.tolist()
        df['model_prediction'] = np.argmax(gather_pred, 1).tolist()
        df['label'] = gather_target.tolist()
        
        save_result_dir = os.path.join(config.test_result_dir, config.exp_name)
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
        
        save_csv = os.path.join(save_result_dir , f'acc_[{metric_dict["test_acc"]*100:.2f}].csv')
        df.to_csv(save_csv , index = True)

        # ================== save confusion matrix ====================

        cm = confusion_matrix(gather_target, np.argmax(gather_pred, 1))
        labels = config.confusion_labels
        plt.figure(figsize=(7, 7))
        sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap = 'YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Label')
        
        save_confusion = os.path.join(save_result_dir , f'confusion_matrix_[{metric_dict["test_acc"]*100:.2f}].png')
        plt.savefig(save_confusion)
        
    
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer