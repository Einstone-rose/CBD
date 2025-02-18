import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
# from models.my_model import MyModel
from models.base_model_mae_full import BaseModel
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score
import warnings
from utils.util import set_requires_grad
warnings.filterwarnings("ignore")

def build_ft_model(args, configs, device, checkpoint=None):
    # model = MyModel('finetune', configs.input_channels, configs.seq_len, 128, 4, 4, 128, configs.num_classes)
    model = BaseModel(configs, 'finetune')
    model.to(device)
    model_dict = model.state_dict()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.ft_lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if checkpoint is not None:
        pretrained_dict = checkpoint["model_state_dict"]
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model, model_optimizer, model_scheduler


def Trainer(train_dl, val_dl, test_dl, device, logger, args, configs, experiment_log_dir, start_epoch, load_path):
    # Finetune all the pretrain results
    logger.debug("Linear-training started ....")
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)
    best_performance = None
    chkpoint = None
    acc_list = []
    for epoch in range(start_epoch, configs.num_epoch + 1, 5):
        logger.debug(f"pretrain epoch {epoch} starting lt ....")
        local_best_performance = None
        ckp_path = f'{load_path}/ckp_ep{epoch}.pt'
        chkpoint = torch.load(ckp_path)
        ft_model, ft_model_optimizer, ft_model_scheduler = build_ft_model(args, configs, device, checkpoint=chkpoint)
        should_save = False
        for ft_epoch in range(1, configs.finetune_epoch + 1):
            ft_loss, ft_acc, F1 = model_finetune(ft_model, train_dl, device, ft_model_optimizer)
            valid_loss, val_acc, _ = model_test(ft_model, val_dl, device)
            ft_model_scheduler.step(valid_loss)  # use valid loss to update lr
            if ft_epoch % 100 == 0:
                logger.debug(f'{ft_epoch}: Train Loss     : {ft_loss:.4f}\t | \tTrain Accuracy     : {ft_acc:2.4f}\t'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {val_acc:2.4f}')
            if ft_epoch % 5 == 0:
                # Test
                # logger.debug(f'Epoch : {ft_epoch}\t | \t  finetune Loss: {ft_loss:.4f}\t | \tAcc: {ft_acc:2.4f}\t | \tF1: {F1:0.4f}')
                test_loss, test_acc, performance = model_test(ft_model, test_dl, device)
                # print(f'ft epoch: {ft_epoch}, test_acc: {test_acc}')
                if local_best_performance is None:
                    local_best_performance = performance
                else:
                    if performance > local_best_performance:
                        local_best_performance = performance
                if best_performance is None:
                    best_performance = performance
                else:
                    if performance[0] > best_performance[0]:
                        best_performance = performance
                        should_save = True
                        logger.debug(
                            'EP%s - Better Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
                                                            ft_epoch, performance[0], performance[1], performance[2], performance[3]))
                        chkpoint = {'epoch': ft_epoch, 'train_loss': test_loss,
                                            'model_state_dict': ft_model.state_dict()}
        if should_save:
            torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_best_ep_{epoch}_acc{best_performance[0]:.4f}.pt'))
        # if epoch % 20 == 0:
        #     torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ft_ckp_best_ep_{epoch}.pt'))
        logger.debug(f"pretrain epoch {epoch} best acc: {local_best_performance[0]:.4f}")
        acc_list.append(local_best_performance[0])
    logger.debug("Linear-training ended ....")
    logger.debug("=" * 100)
    logger.debug('EP%s - Best Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
                epoch, best_performance[0], best_performance[1], best_performance[2], best_performance[3]))
    logger.debug("=" * 100)

    return best_performance, acc_list
            

def model_finetune(model, val_dl, device, model_optimizer):
    model.train()
    total_loss = []
    total_acc = []
    criterion = nn.CrossEntropyLoss()

    for data, labels in val_dl:
        data, labels = data.float().to(device), labels.long().to(device)
        pred, _ = model(data)  # [bs, class_num]
        loss = criterion(pred, labels)

        acc_bs = labels.eq(pred.detach().argmax(dim=1)).float().mean()
        pred_numpy = pred.detach().cpu().numpy()

        total_acc.append(acc_bs)
        total_loss.append(loss.item())

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    # F1 score
    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

    # average loss and acc
    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc

    return total_loss, total_acc, F1


def model_test(ft_model, test_dl, device):
    ft_model.eval()
    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            pred_test, _ = ft_model(data)
            loss = criterion(pred_test, labels)
            acc_bs = labels.eq(pred_test.detach().argmax(dim=1)).float().mean()
            pred_numpy = pred_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()

            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_acc.append(acc_bs)
            total_loss.append(loss.item())

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100]

    return total_loss, total_acc, performance
