import os
import sys
sys.path.append("..")
import torch
from utils.my_loss import TimeRebuildLoss, MyFocalFrequencyLoss
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def Trainer(model, model_optimizer, train_dl, device, logger, args, configs, 
            experiment_log_dir, seed, start_epoch, training_mode):
    logger.debug("Pre-training started ....")
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=configs.num_epoch, eta_min=configs.lr/10)
    all_loss = []
    # Pre-training
    for epoch in range(start_epoch, configs.num_epoch + 1):
        total_loss, total_ts_loss, total_focal_freq_loss, total_ts_loss_freq, total_freq_loss_freq = model_pretrain(model, model_optimizer, model_scheduler, train_dl, configs, args, device)
        all_loss.append(total_loss)
        logger.debug(f"Epoch {epoch}/{configs.num_epoch}: loss: {total_loss}, ts loss: {total_ts_loss}, fq loss: {total_focal_freq_loss}, ts_freq loss: {total_ts_loss_freq}, fq_freq loss: {total_freq_loss_freq}\n")
        chkpoint = {'seed': seed, 'epoch': epoch, 'train_loss': total_loss, 'model_state_dict': model.state_dict(), 'model_optimizer': model_optimizer.state_dict(), 'model_scheduler': model_scheduler.state_dict()}
        if epoch >= 10 and epoch % 5 == 0:
            torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_ep{epoch}.pt'))
    return all_loss


def model_pretrain(model, model_optimizer, model_scheduler, train_loader, configs, args, device):
    total_loss = []
    total_ts_loss = []
    total_focal_freq_loss = []
    total_ts_loss_freq = []
    total_freq_loss_freq = []
    model.train()
    lam = configs.lam

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.float().to(device)  # [batch_size, nvars, seq_len]
        labels = labels.float().to(device)

        rec_data, rec_freq = model(data)
        ts_loss = get_time_rebuild_loss(device=device, args=args, origin_data=data, recon_data=rec_data.transpose(1, 2))
        focal_freq_loss = get_freq_rebuild_loss(data, rec_data.transpose(1, 2))
        # loss_spa = ts_loss * ts_ratio + focal_freq_loss * (1 - ts_ratio)
        loss_spa = ts_loss + focal_freq_loss
        # loss_spa = ts_loss

        spatial_x = torch.fft.irfft(rec_freq, norm='ortho').abs()
        ts_loss_freq = get_time_rebuild_loss(device=device, args=args, origin_data=data, recon_data=spatial_x)
        focal_freq_loss_freq = get_freq_rebuild_loss(data, rec_freq, is_freq=True)
        loss_freq = ts_loss_freq + focal_freq_loss_freq
        
        loss = loss_spa + loss_freq * lam
        # loss = loss_spa

        # if batch_idx % 30 == 0:
        #     currentTime = datetime.now().time()
        #     print(f'time: {currentTime}, batch: {batch_idx}, loss: {loss}, ts loss: {ts_loss}, fq loss: {focal_freq_loss}')

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        # for name, param in model.named_parameters():
        #     want_list = ["deconv_block1.0.weight", "deconv_block2.0.weight", "deconv_block3.0.weight"]
        #     if param.grad is not None and name in want_list:
        #         print(f'Gradient of {name}: {param.grad}')

        total_loss.append(loss.item())
        total_ts_loss.append(ts_loss.item())
        total_focal_freq_loss.append(focal_freq_loss.item())
        total_ts_loss_freq.append(ts_loss_freq.item())
        total_freq_loss_freq.append(focal_freq_loss_freq.item())

    total_loss = torch.tensor(total_loss).mean()
    total_ts_loss = torch.tensor(total_ts_loss).mean()
    total_focal_freq_loss = torch.tensor(total_focal_freq_loss).mean()
    total_ts_loss_freq = torch.tensor(total_ts_loss_freq).mean()
    total_freq_loss_freq = torch.tensor(total_freq_loss_freq).mean()

    # model_scheduler.step()
    return total_loss, total_ts_loss, total_focal_freq_loss, total_ts_loss_freq, total_freq_loss_freq


def get_freq_rebuild_loss(origin_data, recon_data, is_freq=False):
    rb_loss = MyFocalFrequencyLoss()
    return rb_loss(recon_data, origin_data, is_freq=is_freq)

def get_time_rebuild_loss(device, args, origin_data, recon_data):
    # all shape is [batch_size, nvars, seq_len]
    rb_loss = TimeRebuildLoss(device=device, args=args)
    return rb_loss(origin_data, recon_data)


