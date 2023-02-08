import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from models.mil_net import MILNetImageOnly, MILNetWithClinicalData
from models.backbones.backbone_builder import BACKBONES
from utils.utils import *
from utils.recorder import Recoder
from breast import BreastDataset_infer
import random
import warnings
import pandas as pd


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
    parser.add_argument("--model_path", required=True)

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
    parser.add_argument("--merge_method", choices=["max", "mean", "not_use"], default="mean") # 최종적으로 bag 에 대한 class를 정의할떄, 각 패치들의 결과 score 를 Mean 해서 낼지, max로 할지 설정
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
    inference_dataset = BreastDataset_infer(args.test_json_path, args.data_dir_path, args.clinical_data_path, is_preloading=args.preloading)
    inference_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return inference_loader

if __name__ == "__main__":
    args = parser_args()

    # init setting
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)

    # init dataloader
    inference_loader = init_dataloader(args)

    # load model
    if args.clinical_data_path:
        model = MILNetWithClinicalData(num_classes=args.num_classes, backbone_name=args.backbone, clinical_data_size=args.clinical_data_size) # init
    else:
        model = MILNetImageOnly(num_classes=args.num_classes, backbone_name=args.backbone)

    # load pth
    print('INFERENCE ... {}'.format(args.model_path))
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    # init training function
    if args.num_classes > 2:
        print(f"multiple classification")
        main_fun = train_val_test_multi_class
    else:
        print(f"binary classification")
        main_fun = train_val_test_binary_class

    # inference
    id_list, predicted_label_list, score_list, bag_num_list = main_fun("inference", 0, model, inference_loader, None, None, None, args.merge_method)

    # save
    inf_dir = os.path.join(args.log_dir_path, 'inference')
    os.makedirs(inf_dir, exist_ok=True)
    
    inf_df = pd.DataFrame()
    inf_df["ID"] = id_list
    inf_df["N_category"] = predicted_label_list
    inf_df["score"] = score_list
    inf_df["bag_num"] = bag_num_list

    inf_df.to_csv(os.path.join(inf_dir, 'submission.csv'))    