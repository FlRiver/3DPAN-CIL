import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
# from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
# from tqdm import tqdm
import logging
import copy

from Point_MCIL_model import Point_MCIL
from Point_MCIL_data import ModelNetDataLoader, IncrementalDataSplitter, PointcloudScaleAndTranslate
from Point_MCIL_utils import Logger, set_random_seed, write_lists_to_txt, save_model, save_args

from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
 


# CUDA_VISIBLE_DEVICES=2


train_transforms = transforms.Compose(
        [
            PointcloudScaleAndTranslate(),
        ]
    )

test_transforms = transforms.Compose(
        [
            PointcloudScaleAndTranslate(),
        ]
    )


"""Parameters"""
def get_args():
    parser = argparse.ArgumentParser()
    # main set
    parser.add_argument('--experiment_name', type=str, default='test1', help='experiment name')
    parser.add_argument('--experiment_path', type=str, default='./checkpoint/', help='path to save checkpoint (default: ckpt)')
    parser.add_argument('--epochs', type=int, default=300,  help='number of epoch in training')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size in training')
    parser.add_argument('--tasks', type=int, default=10, help='tasks num')
    parser.add_argument('--split_way',type=str, default='avg', help='dataset split way (avg/half)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='decay rate')
    parser.add_argument('--grad_norm_clip', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--checkpoint_name', type=str, default='acc', help='acc/loss/last')
    
    # model set
    parser.add_argument('--main_model', default='Point-MCIL', help='main model name')
    parser.add_argument('--mask_ratio', type=int, default=0.5, help='mask ratio')
    parser.add_argument('--base_n_class', type=int, default=4, help='Number of classes in the basic stage')
    parser.add_argument('--optim', type=str, default="adamw", help='sgd/adamw/adam')
    parser.add_argument('--sched', type=str, default="cosLR", help='lambdaLR/cosLR/stepLR')

    # data set
    parser.add_argument('--dataset', type=str, default="modelnet40", help='modelnet10/modelnet40/fewshot/shapenet55')
    parser.add_argument('--workers', default=8, type=int, help='workers')
    parser.add_argument('--num_points', type=int, default=2048, help='point number')


    return parser.parse_args()


def main():

    args = get_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.experiment_path = args.experiment_path + args.experiment_name

    set_random_seed(seed=args.seed)
    os.makedirs(os.path.join(args.experiment_path, "pseudo_sample"), exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s [%(levelname)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=os.path.join(args.experiment_path, "log_output.txt"),
                filemode='w')
    
    TrainDataLoaders = []
    TestDataLoaders = []
    class_splites_test = []

 
    if args.dataset == "modelnet40" or args.dataset == "modelnet10":
        args.dataset_path = "/home/xyj/data/modelnet"
        Incremental = IncrementalDataSplitter(args.dataset_path, args.dataset, args.split_way, args.tasks)
        class_splites, class_cls = Incremental.split_class()
 
        args.base_n_class = len(class_splites[0])

        for t in range(args.tasks):
            dataloader_train = ModelNetDataLoader(args.dataset_path, class_splites[t], class_cls, split="train")

            for _ in class_splites[t]:            
                class_splites_test.append(_)
            dataloader_test = ModelNetDataLoader(args.dataset_path, class_splites_test, class_cls, split="test")
    
            TrainDataLoaders.append(torch.utils.data.DataLoader(dataloader_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers))
            TestDataLoaders.append(torch.utils.data.DataLoader(dataloader_test, batch_size=args.batch_size, num_workers=args.workers))
    elif args.dataset == "shapenet55":
        args.dataset_path = "/home/xyj/data/ShapeNet55-34"
    elif args.dataset == "fewshot":
        args.dataset_path = "/home/xyj/data/ModelNetFewshot"
    else:
        raise FileNotFoundError(f"The dataset at '{args.dataset}' does not exist.")


    logging.info(f'='*66)
    logging.info(f'Args parameters:')
    for arg_name, arg_value in vars(args).items():
        logging.info(f'{arg_name}: {arg_value}')
    save_args(args)

    logging.info(f'='*66)
    logging.info(f'Classes for each stage:')
    for t in range(args.tasks):
        logging.info(f'The {t+1}-th stage: {class_splites[t]}')
    logging.info(f'='*66)

    model = Point_MCIL(n_classes=args.base_n_class, 
                       mask_ratio=args.mask_ratio,
                       ).to(args.device)

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-4) 
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-4)
    else:
        raise NotImplementedError()
    
    if args.sched == "cosLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    elif args.sched == "stepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif args.sched == "lambdaLR":
        lambda_func = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    else:
        raise NotImplementedError()



    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    logging.info(f"model parameters: {str(num_params)}")
    logging.info("="*66)

    PseudoSample = {}


    logger_train = Logger(os.path.join(args.experiment_path, 'log_train.txt'))
    logger_train.set_names(['Task-num', 'Epoch-num', 'Learn-rate', 'Train-acc', 'Valid-acc'])

    model.zero_grad()
    for task in range(args.tasks):
        best_test_acc = 0.
        best_train_acc = 0.
        best_test_loss = float("inf")
        best_train_loss = float("inf")

        if task !=0:
 
            model_old = copy.deepcopy(model)
            model_old.to(args.device)
            model.update_class(len(class_splites[task]))
            model.to(args.device)

        for epoch in range(args.epochs):
            logging.info('Task(%d/%s) - Epoch(%d/%s) Learning Rate %s:' % (task+1, args.tasks, epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
            train_out = train(args, epoch, model, TrainDataLoaders[task], optimizer, scheduler)
            ps = model.memory_bank()
            model.class_pseudo = {}
            logging.info("="*66)
            test_out = validate(args, epoch, model, TestDataLoaders[task])
            logging.info("="*66)

            if test_out["acc_test"] > best_test_acc:
                acc_is_best = True
            else:
                acc_is_best = False

            
            if train_out["loss_train"] < best_train_loss:
                loss_is_best = True
            else:
                loss_is_best = False


            best_test_acc = test_out["acc_test"] if (test_out["acc_test"] > best_test_acc) else best_test_acc
            best_train_acc = train_out["acc_train"] if (train_out["acc_train"] > best_train_acc) else best_train_acc
            best_test_loss = test_out["loss_test"] if (test_out["loss_test"] < best_test_loss) else best_test_loss
            best_train_loss = train_out["loss_train"] if (train_out["loss_train"] < best_train_loss) else best_train_loss
            save_model(
                model, epoch, T=task+1,
                path=args.experiment_path,
                acc=test_out["acc_test"],
                loss_is_best=loss_is_best,
                acc_is_best=acc_is_best,
                best_test_acc=best_test_acc,
                best_train_acc=best_train_acc,
                best_test_loss=best_test_loss,
                best_train_loss=best_train_loss,
                optimizer=optimizer.state_dict()
            )
            if loss_is_best is True:
                for key, value in ps.items():
                    PseudoSample[key] = value

            logger_train.append([task+1, epoch+1, optimizer.param_groups[0]['lr'], train_out["acc_train"], test_out["acc_test"]])


        logging.info("="*66)
        logging.info(f"Best Result task: {task+1}")
        logging.info(f"Best Train loss: {best_train_loss}")
        logging.info(f"Best Test loss: {best_test_loss}")
        logging.info(f"Best Train acc: {best_train_acc}")
        logging.info(f"Best Test acc: {best_test_acc}")
        logging.info("="*66)
    logger_train.close()

    test(args, TestDataLoaders, class_splites, args.checkpoint_name)



def final_distillation_loss(student_logits, teacher_logits, T=20):

    teacher_probs = nn.functional.softmax(teacher_logits / T, dim=1)
    student_probs = nn.functional.softmax(student_logits / T, dim=1)
    knowledge_distillation_loss = nn.KLDivLoss()(student_probs, teacher_probs)
    
    return knowledge_distillation_loss


def layerwise_distillation_loss(student_intermediate_outputs, teacher_intermediate_outputs, loss_weights=None):

    if loss_weights is None:
        loss_weights = [1/len(student_intermediate_outputs)] * len(student_intermediate_outputs)
    
    total_layerwise_loss = 0
    for i, (s_out, t_out) in enumerate(zip(student_intermediate_outputs, teacher_intermediate_outputs)):
        mse_loss = nn.MSELoss()(s_out, t_out)
        total_layerwise_loss += loss_weights[i] * mse_loss
        
    return total_layerwise_loss


def REandCE_loss(re, gt, xp, xl):
    cd1 = ChamferDistanceL1().cuda()(re, gt)
    cd2 = ChamferDistanceL2().cuda()(re, gt)
    cls = nn.CrossEntropyLoss()(xp, xl)
    
    return cd1, cd2, cls


def train(args, epoch, net, trainloader, optimizer, scheduler):
    net.train()
    every_loss = 0
    every_accuracy = 0
    flag = 0
    for point, label in trainloader:
        point = point.to(args.device)
        point = train_transforms(point)

        label = label.to(args.device).reshape(label.shape[0])
    return {
        "loss_train": float("%.4f" % (every_loss/flag)),
        "acc_train": float("%.4f" % (100.*every_accuracy/flag)),
    }


if __name__ == '__main__':
    main()
