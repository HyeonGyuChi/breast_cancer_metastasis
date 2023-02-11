import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from models.mil_net import MILNetImageOnly, MILNetWithClinicalData
from models.backbones.backbone_builder import BACKBONES
from utils.utils import *
from utils.recorder import Recoder
from breast import BreastDataset
import random
import warnings

import wandb
wandb.init(project="breast_cancer_metastasis", entity="hyeongyuc96")

def parser_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--train_json_path", required=True)
    parser.add_argument("--val_json_path", required=True)
    parser.add_argument("--test_json_path", required=True)
    parser.add_argument("--data_dir_path", default="./dataset")
    parser.add_argument("--clinical_data_path")
    parser.add_argument("--clinical_data_size", type=int)
    parser.add_argument("--preloading", action="store_true")
    parser.add_argument("--num_classes", type=int, choices=[2, 3], default=2)

    # model
    parser.add_argument("--backbone", choices=BACKBONES, default="vgg16_bn")

    # optimizer
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="SGD")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--l1_weight", type=float, default=0)

    # output
    parser.add_argument("--log_dir_path", default="./logs")
    parser.add_argument("--log_name", required=True)
    parser.add_argument("--save_epoch_interval", type=int, default=10)

    # other
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--train_stop_auc", type=float, default=0.98)
    parser.add_argument("--merge_method", choices=["max", "mean", "not_use"], default="mean")
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--num_workers", type=int, default=6)

    args = parser.parse_args()

    return args


def init_output_directory(log_dir_path, log_name):
    tensorboard_path = os.path.join(log_dir_path, log_name, "tensorboard")
    checkpoint_path = os.path.join(log_dir_path, log_name, "checkpoint")
    xlsx_path = os.path.join(log_dir_path, log_name, "xlsx")

    for path in [tensorboard_path, checkpoint_path, xlsx_path]:
        os.makedirs(path, exist_ok=True)
        print(f"init path {path}")

    return tensorboard_path, checkpoint_path, xlsx_path


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_optimizer(args, model):
    if args.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError


def init_dataloader(args):
    train_dataset = BreastDataset(args.train_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_dataset = BreastDataset(args.val_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    test_dataset = BreastDataset(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    args = parser_args()
    
    # wandb setup
    wandb.config.update(args)

    # init setting
    warnings.filterwarnings("ignore")
    tensorboard_path, checkpoint_path, xlsx_path = init_output_directory(args.log_dir_path, args.log_name)
    seed_everything(args.seed)

    # init logger
    writer = SummaryWriter(log_dir=tensorboard_path, flush_secs=10)
    train_recoder = Recoder(xlsx_path, "train")
    val_recoder = Recoder(xlsx_path, "val")
    test_recoder = Recoder(xlsx_path, "test")

    # init dataloader
    train_loader, val_loader, test_loader = init_dataloader(args)

    # init model
    if args.clinical_data_path:
        model = MILNetWithClinicalData(num_classes=args.num_classes, backbone_name=args.backbone, clinical_data_size=args.clinical_data_size)
    else:
        model = MILNetImageOnly(num_classes=args.num_classes, backbone_name=args.backbone)
    model = model.cuda()

    # init optimizer and lr scheduler
    optimizer = init_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20, T_mult=2)

    # init training function
    if args.num_classes > 2:
        print(f"multiple classification")
        main_fun = train_val_test_multi_class
    else:
        print(f"binary classification")
        main_fun = train_val_test_binary_class

    # training
    best_auc = 0
    best_epoch = 0
    for epoch in range(1, args.epoch + 1):
        scheduler.step()
    
        train_metrics, train_loss, _, _ = main_fun("train", epoch, model, train_loader, optimizer, train_recoder, writer, args.merge_method)
        val_metrics, val_loss, _, _ = main_fun("val", epoch, model, val_loader, None, val_recoder, writer, args.merge_method)
        test_metrics, test_loss, label_list, score_list = main_fun("test", epoch, model, test_loader, None, test_recoder, writer, args.merge_method)
        predicted_label_list = [1 if score >= 0.5 else 0 for score in score_list]
        
        # wandb log
        wandb.log({
            "epoch": epoch,
        }, commit=False)
        
        wandb.log({
            'train_loss': train_loss,
            'train_auc': train_metrics['auc'],
            'train_acc': train_metrics['acc'],
            'train_sens(recall)': train_metrics['sens'],
            'train_spec': train_metrics['spec'],
            'train_ppv(precision)': train_metrics['ppv'],
            'train_npv': train_metrics['npv'],
            'train_f1': train_metrics['f1'],
        }, commit=False)
        
        wandb.log({
            'val_loss': val_loss,
            'val_auc': val_metrics['auc'],
            'val_acc': val_metrics['acc'],
            'val_sens(recall)': val_metrics['sens'],
            'val_spec': val_metrics['spec'],
            'val_ppv(precision)': val_metrics['ppv'],
            'val_npv': val_metrics['npv'],
            'val_f1': val_metrics['f1'],
        }, commit=False)
        
        wandb.log({
            'test_loss': test_loss,
            'test_auc': test_metrics['auc'],
            'test_acc': test_metrics['acc'],
            'test_sens(recall)': test_metrics['sens'],
            'test_spec': test_metrics['spec'],
            'test_ppv(precision)': test_metrics['ppv'],
            'test_npv': test_metrics['npv'],
            'test_f1': test_metrics['f1'],
        }, commit=False)
        
        # get only auc    
        train_auc = train_metrics['auc']
        val_auc = val_metrics['auc']
        
        # save best
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            save_checkpoint(model, os.path.join(checkpoint_path, "best.pth"))

        # save model
        if epoch % args.save_epoch_interval == 0:
            save_checkpoint(model, os.path.join(checkpoint_path, f"{epoch}.pth"))
            save_checkpoint(model, os.path.join(checkpoint_path, f"last.pth"))

        print("-" * 120)

        # early stopping
        if train_auc > args.train_stop_auc:
            # score expansion for plot
            new_score = np.zeros((len(score_list), 2))
            for i, lab in enumerate(label_list):
                if lab == 0:
                    new_score[i, 0] = score_list[i]
                    new_score[i, 1] = 1 - score_list[i]
                elif lab == 1:
                    new_score[i, 0] = 1 - score_list[i]
                    new_score[i, 1] = score_list[i]
            
            wandb.log({
                'pr': wandb.plot.pr_curve(np.array(label_list), new_score, labels=['N', 'P'])
            }, commit=False)
            
            wandb.log({
                'roc': wandb.plot.roc_curve(np.array(label_list), new_score, labels=['N', 'P'])
            }, commit=False)
            
            wandb.log({
                'Confusion Matrix': wandb.plot.confusion_matrix\
                    (y_true=np.array(label_list), preds=np.array(predicted_label_list), class_names=['N', 'P'])
            }, commit=True)
            
            print(f"early stopping, epoch: {epoch}, train_auc: {train_auc:.3f} (>{args.train_stop_auc})")
            break
        
        else:
            wandb.log({}, commit=True)

        torch.cuda.empty_cache()

    writer.close()
    print(f"end, best_auc: {best_auc}, best_epoch: {best_epoch}")
