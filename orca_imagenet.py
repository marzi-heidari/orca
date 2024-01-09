import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_imagenet as datasets
import torchvision.transforms as transforms

from orca_cifar import compute_prototype, kmeans, update_prototype, calculate_dis_loss, calculate_gen_loss, \
    compute_unsup_loss, euclidean_dist
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pykeops.torch import LazyTensor
from torch.utils.data import SequentialSampler
from tqdm import tqdm

import models

import utils_u
from diffusion_u import create_diffusion, ema, timestep_sampler
from diffusion_u.timestep_sampler import UniformSampler
from models.resnet_s import SimpleDiscriminator
from models.transformer import Gpt
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, BalancedBatchSampler
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


def train(args, model, device, train_loader, optimizer, epoch, tf_writer, diffusion,
          diffusion_model, diffusion_optimizer, timestep_sampler, prototypes, discriminator, optimizer_d, labeled_len):
    model.train()
    diffusion_model.train()
    discriminator.train()


    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    for ((x, x2), target,idx) in train_loader:

        target=target[:labeled_len]
        x, x2, target = x.to(device), x2.to(device), target.to(device)

        optimizer.zero_grad()
        diffusion_optimizer.zero_grad()
        optimizer_d.zero_grad()
        output, feat = model(x)

        output2, feat_ = model(x2)

        # logits_x_ulb_w, feat2 = model(ux2)

        labeled_len = len(target)
        # Compute context vectors for unlabeled data

        one_hot_labels = F.one_hot(target, args.no_class)
        p1 = compute_prototype(feat[:labeled_len, ], one_hot_labels, args.no_class // 2, gpu=args.gpu)
        pseudo_labels = F.softmax(output2[labeled_len:], dim=1).detach()
        if epoch == 0:
            p2 = kmeans(feat[labeled_len:, :], args.no_class, )[0]
        else:
            # pseudo_labels = calc_pseudo_lables(feat, labeled_len, prototypes)
            p2 = compute_prototype(feat[labeled_len:, :], pseudo_labels, args.no_class, gpu=args.gpu, is_unseen=True, )
        proto = torch.cat((p1, p2[args.no_class // 2:, :].cuda(args.gpu)), dim=0)
        prototypes = update_prototype(prototypes, proto)

        t, vlb_weights = timestep_sampler.sample(args.batch_size, args.gpu)
        t = t.cuda(args.gpu)
        context_vector = torch.zeros((args.batch_size, 2048)).cuda(args.gpu)
        for i in range(labeled_len):
            context_vector[i] = prototypes[target[i]]
        for i in range(labeled_len, args.batch_size):
            context_vector[i] = prototypes[torch.argmax(pseudo_labels[i - labeled_len])]

        diffusion_loss_, denoised_labeld_ = diffusion.training_losses(diffusion_model,
                                                                      feat,
                                                                      context_vector.detach(), t)

        discriminator_loss = calculate_dis_loss(discriminator, diffusion_model, context_vector.detach(), feat,

                                                diffusion, args.gpu)

        discriminator_loss1 = calculate_gen_loss(discriminator, diffusion_model, context_vector.detach(),

                                                 diffusion, args.gpu)

        # logits_ub_s = output[labeled_len:, :]
        # unsup_loss = compute_unsup_loss(pseudo_labels, logits_ub_s, args.gpu, num_classes=args.no_class)
        # bce_loss = calculate_bce_loss(args, output2, labeled_len, F.softmax(output, dim=1), feat, target)
        query_to_proto_distance = euclidean_dist(feat[:labeled_len], prototypes[:50])

        ce_loss = F.cross_entropy(output[:labeled_len], target) + F.cross_entropy(-query_to_proto_distance, target)
        # distances = euclidean_dist(feat[labeled_len:], prototypes)
        # distances2 = euclidean_dist(feat_[labeled_len:], prototypes)
        unsup_loss = compute_unsup_loss(pseudo_labels, output[labeled_len:], args.gpu, num_classes=args.no_class)

        if epoch < 5:
            loss = ce_loss
        else:
            loss = ce_loss + 0.5 * unsup_loss + discriminator_loss1 + discriminator_loss + diffusion_loss_.mean()

        # bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(ce_loss.item(), args.batch_size)
        optimizer.zero_grad()

        diffusion_optimizer.zero_grad()
        optimizer_d.zero_grad()

        loss.backward()
        # discriminator_loss.backward()
        optimizer.step()
        diffusion_optimizer.step()
        optimizer_d.step()

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)
    return prototypes


best_acc = 0
best_seen_acc = 0
best_unseen_acc = 0
best_epoch = 0


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer, diffusion, diffusion_model, prototypes):
    global best_acc, best_seen_acc, best_unseen_acc, best_epoch
    model.eval()
    diffusion_model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            # t = torch.ones(output.shape[0], dtype=torch.long).cuda(args.gpu)
            # t = t.cuda(args.gpu)
            # denoised_labeld = diffusion.p_sample(diffusion_model,
            #                                      output, F.softmax(output, dim=1), t)
            #
            # output = denoised_labeld["pred_xstart"]
            # query_to_proto_distance = euclidean_dist(feat, prototypes)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    if best_acc < overall_acc:
        best_acc = overall_acc
        best_unseen_acc = unseen_acc
        best_seen_acc = seen_acc
        best_epoch = epoch
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    print('Test unseen nmi {:.4f}, mean uncertainty {:.4f}'.format(unseen_nmi, mean_uncert))
    print('Best overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, best epoch {}\n'.format(best_acc, best_seen_acc,
                                                                                                best_unseen_acc,
                                                                                                best_epoch))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert


parser = argparse.ArgumentParser(
    description='orca',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='imagenet100', help='dataset setting')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60])
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset_root', default='../../data/imagenet/train/', type=str)
parser.add_argument('--exp_root', type=str, default='./results/')
parser.add_argument('--labeled-num', default=50, type=int)
parser.add_argument('--labeled-ratio', default=0.5, type=float)
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--no_class', default=100, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--name', type=str, default='debug')


if __name__ == "__main__":

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    args.savedir = args.savedir + '/'

    model = models.resnet50(num_classes=args.no_class)
    state_dict = torch.load('./pretrained/simclr_imagenet_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if 'fc' not in name and 'layer4' not in name:
            param.requires_grad = False

    model = model.to(device)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_label_set = datasets.ImageNetDataset(root=args.dataset_root,
                                               anno_file='./data/ImageNet100_label_{}_{:.1f}.txt'.format(
                                                   args.labeled_num, args.labeled_ratio),
                                               transform=TransformTwice(transform_train))
    train_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root,
                                                 anno_file='./data/ImageNet100_unlabel_{}_{:.1f}.txt'.format(
                                                     args.labeled_num, args.labeled_ratio),
                                                 transform=TransformTwice(transform_train))
    concat_set = datasets.ConcatDataset((train_label_set, train_unlabel_set))
    labeled_idxs = range(len(train_label_set))
    unlabeled_idxs = range(len(train_label_set), len(train_label_set) + len(train_unlabel_set))
    batch_sampler = datasets.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                                   int(args.batch_size * len(train_unlabel_set) / (
                                                               len(train_label_set) + len(train_unlabel_set))))

    test_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root,
                                                anno_file='./data/ImageNet100_unlabel_50_0.5.txt',
                                                transform=transform_test)

    train_loader = torch.utils.data.DataLoader(concat_set, batch_sampler=batch_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_unlabel_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=8)
    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    discriminator = SimpleDiscriminator(2048)
    discriminator = discriminator.cuda(args.gpu)
    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False
    diffusion_model = Gpt(prototype_sizes=2048,
                          predict_xstart=False,
                          max_freq_log2=14,
                          num_frequencies=128,
                          n_embd=2048,
                          encoder_depth=1,
                          n_layer=1,
                          n_head=1,
                          len_input=3,
                          attn_pdrop=0.0,
                          resid_pdrop=0.0,
                          embd_pdrop=0.0).cuda(args.gpu)
    diffusion = create_diffusion(learn_sigma=False, predict_xstart=False,
                                 noise_schedule='linear', steps=50)
    timestep_sampler = UniformSampler(diffusion)

    ema_helper = ema.EMAHelper(mu=0.9999)
    ema_helper.register(diffusion_model)
    optimizer_diffusion, lr_scheduler_diffusion = utils_u.make_optimizer(
        diffusion_model.parameters(),
        'sgd', lr=1.e-3, weight_decay=5.e-4, milestones=[30, 50, 80])
    model = model.cuda(args.gpu)
    diffusion_model.cuda(args.gpu)
    # Set the optimizer

    optimizer_d = optim.SGD(discriminator.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)
    prototypes = torch.zeros((args.no_class, 2048)).cuda(args.gpu)

    for epoch in tqdm(range(args.epochs)):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer, diffusion,
                           diffusion_model, prototypes)
        prototypes = train(args, model, device, train_loader, optimizer, epoch, tf_writer,
                           diffusion,
                           diffusion_model, optimizer_diffusion, timestep_sampler, prototypes, discriminator,
                           optimizer_d,batch_sampler.primary_batch_size)
        scheduler.step()
        lr_scheduler_diffusion.step()
