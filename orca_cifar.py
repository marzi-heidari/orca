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
import open_world_cifar as datasets
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


def update_prototype(old_proto, prototypes, alpha=0.9):
    for i in range(old_proto.shape[0]):
        if old_proto[i].sum() == 0:
            old_proto[i] = prototypes[i]

    prototypes = alpha * old_proto + (1 - alpha) * prototypes
    return prototypes


def calculate_dis_loss(discriminator, diffusion_model, prototypes, feat, pseudo_labels, diffusion, gpu):
    criterion = nn.BCEWithLogitsLoss()

    # Forward pass for real data
    real_outputs = discriminator(feat.detach())
    real_labels = torch.ones_like(real_outputs)  # Real data label is 1
    real_loss = criterion(real_outputs, real_labels)
    with torch.no_grad():
        z_0 = diffusion.p_sample_loop(diffusion_model, prototypes, prototypes.shape, device=f'cuda:{gpu}')
    # Forward pass for fake data
    fake_outputs = discriminator(z_0.detach())
    fake_labels = torch.zeros_like(fake_outputs)  # Fake data label is 0
    fake_loss = criterion(fake_outputs, fake_labels)

    # Total loss
    return real_loss + fake_loss


def calculate_gen_loss(discriminator, diffusion_model, prototypes, diffusion, gpu):
    criterion = nn.BCEWithLogitsLoss()

    z_0 = diffusion.p_sample_loop(diffusion_model, prototypes.detach(), prototypes.shape, device=f'cuda:{gpu}')
    # Forward pass for fake data
    with torch.no_grad():
        fake_outputs = discriminator(z_0)
    fake_labels = torch.ones_like(fake_outputs)  # Fake data label is 1
    fake_loss = criterion(fake_outputs, fake_labels)

    # Total loss
    return fake_loss


def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer, diffusion,
          diffusion_model, diffusion_optimizer, timestep_sampler, prototypes, discriminator, optimizer_d):
    model.train()
    diffusion_model.train()
    discriminator.train()

    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    for ((x, x2), target) in train_label_loader:
        ((ux, ux2), unlabeled_target_) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        ux, ux2 = ux.to(device), ux2.to(device)
        optimizer.zero_grad()
        diffusion_optimizer.zero_grad()
        optimizer_d.zero_grad()
        output, feat = model(x)

        logits_x_ulb_w, feat2 = model(ux2)

        labeled_len = len(target)
        # Compute context vectors for unlabeled data

        pseudo_labels = F.softmax(logits_x_ulb_w, dim=1).detach()
        one_hot_labels = F.one_hot(target, args.no_class)
        p1 = compute_prototype(feat[:labeled_len, ], one_hot_labels, args.no_class // 2, gpu=args.gpu)
        logits_x_lb = output[:labeled_len, :]

        p2 = compute_prototype(feat[labeled_len:, :], pseudo_labels, args.no_class, gpu=args.gpu)
        proto = torch.cat((p1, p2[args.no_class // 2:, :]), dim=0)
        prototypes = update_prototype(prototypes, proto)
        t, vlb_weights = timestep_sampler.sample(labeled_len, args.gpu)
        t = t.cuda(args.gpu)
        labeled_context = torch.zeros((labeled_len, 512)).cuda(args.gpu)
        for i in range(labeled_len):
            labeled_context[i] = prototypes[target[i]]

        diffusion_loss_, denoised_labeld_ = diffusion.training_losses(diffusion_model,
                                                                      feat[:labeled_len, :].detach(),
                                                                      labeled_context.detach(), t)

        discriminator_loss = calculate_dis_loss(discriminator, diffusion_model, prototypes[50:], feat[labeled_len:],
                                                pseudo_labels,
                                                diffusion, args.gpu)

        discriminator_loss1 = calculate_gen_loss(discriminator, diffusion_model, prototypes[50:],

                                                 diffusion, args.gpu)

        logits_ub_s = output[labeled_len:, :]
        # unsup_loss = compute_unsup_loss(pseudo_labels, logits_ub_s, args.gpu, num_classes=args.no_class)

        query_to_proto_distance = euclidean_dist(feat[:labeled_len], prototypes[:50])
        # ce_loss = F.cross_entropy(logits_x_lb, target)
        ce_loss = F.cross_entropy(-query_to_proto_distance, target)
        distances = euclidean_dist(feat[labeled_len:], prototypes)
        distances2 = euclidean_dist(feat2, prototypes)
        unsup_loss = F.mse_loss(-distances, -distances2)
        prob = F.softmax(-distances, dim=1)
        entropy_loss = entropy(torch.mean(prob, 0))

        loss = - entropy_loss + ce_loss + diffusion_loss_.mean()+unsup_loss  - discriminator_loss1 + discriminator_loss

        # bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
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


def calculate_bce_loss(args, output2, labeled_len, prob, feat, target):
    prob2 = F.softmax(output2, dim=1)

    feat_detach = feat.detach()
    feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
    cosine_dist = torch.mm(feat_norm, feat_norm.t())
    bce = nn.BCELoss()
    pos_pairs = []
    target_np = target.cpu().numpy()
    # label part
    for i in range(labeled_len):
        target_i = target_np[i]
        idxs = np.where(target_np == target_i)[0]
        if len(idxs) == 1:
            pos_pairs.append(idxs[0])
        else:
            selec_idx = np.random.choice(idxs, 1)
            while selec_idx == i:
                selec_idx = np.random.choice(idxs, 1)
            pos_pairs.append(int(selec_idx))
    # unlabel part
    unlabel_cosine_dist = cosine_dist[labeled_len:, :]
    vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
    pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
    pos_pairs.extend(pos_idx)
    pos_prob = prob2[pos_pairs, :]
    pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
    ones = torch.ones_like(pos_sim)
    bce_loss = bce(pos_sim, ones)
    return bce_loss


best_acc = 0
best_seen_acc = 0
best_unseen_acc = 0


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer, diffusion, diffusion_model, prototypes):
    global best_acc, best_seen_acc, best_unseen_acc
    model.eval()
    diffusion_model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            # t = torch.ones(output.shape[0], dtype=torch.long).cuda(args.gpu)
            # t = t.cuda(args.gpu)
            # denoised_labeld = diffusion.p_sample(diffusion_model,
            #                                      output, F.softmax(output, dim=1), t)
            #
            # output = denoised_labeld["pred_xstart"]
            query_to_proto_distance = euclidean_dist(feat, prototypes)
            prob = -query_to_proto_distance
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
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    print('Test unseen nmi {:.4f}, mean uncertainty {:.4f}'.format(unseen_nmi, mean_uncert))
    print('Best overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}\n'.format(best_acc, best_seen_acc,
                                                                                 best_unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert


def compute_unsup_loss(denoised_soft_labels, logits_x_ulb_s, gpu, num_classes=10):
    # Find the pseudo-labels and their confidence
    pseudo_labels = torch.argmax(denoised_soft_labels, dim=1)
    max_probs, _ = torch.max(denoised_soft_labels, dim=1)

    # Select samples where the confidence is above the threshold
    confident_samples = (max_probs > 0.5) & (pseudo_labels > 49)
    # confident_samples =
    confident_labels = pseudo_labels[confident_samples]

    # Balance the set
    unique_labels, counts = confident_labels.unique(return_counts=True)
    if len(confident_labels) == 0:
        return torch.tensor(0.0).cuda(gpu)

    # min_samples = counts.min()

    # balanced_indices = torch.cat(
    #     [confident_samples.nonzero()[confident_labels == label][:min_samples] for label in unique_labels])

    # Compute loss only on the balanced subset
    balanced_logits = logits_x_ulb_s[confident_samples]
    balanced_labels = pseudo_labels[confident_samples]
    # balanced_labels = torch.where(balanced_labels < 5, balanced_labels + 5, balanced_labels)

    # Assuming a standard cross-entropy loss for simplicity
    loss = F.cross_entropy(balanced_logits, balanced_labels)
    return loss


@torch.no_grad()
def compute_prototype(feat, targets, num_classes, is_unseen=False, tau=0.0, gpu=0):
    """
    Compute prototypes for given data.

    Args:
    feat (Tensor): Batch of data (either features or inputs).
    targets (Tensor): Corresponding targets or predicted labels.
    is_unseen (bool): Flag to indicate if the data is for unseen classes.
    tau (float, optional): Threshold for filtering predictions in unseen classes.

    Returns:
    Tensor: Prototype vector.
    """

    if is_unseen and tau is not None:
        # For unseen classes in unlabeled data
        valid_indices = torch.max(targets, dim=1).values > tau
        feat = feat[valid_indices]
        targets = targets[valid_indices]

    # Compute prototypes
    target_index = torch.argmax(targets, dim=1)
    clasees = torch.unique(target_index)

    # assert len(clasees) == num_classes
    prototypes = torch.zeros((num_classes, feat.size(1))).cuda(gpu)

    for idx, cls in enumerate(clasees):
        cls_mask = target_index == cls
        cls_data = feat[cls_mask]
        prototype = torch.mean(cls_data, dim=0)
        prototypes[cls] = prototype.detach()

    return prototypes


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--no_class', default=100, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N',
                        help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                    labeled_ratio=args.labeled_ratio, download=True,
                                                    transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                      labeled_ratio=args.labeled_ratio, download=True,
                                                      transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                      unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                             labeled_ratio=args.labeled_ratio, download=True,
                                             transform=datasets.dict_transform['cifar_test'],
                                             unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                     labeled_ratio=args.labeled_ratio, download=True,
                                                     transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                       labeled_ratio=args.labeled_ratio, download=True,
                                                       transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                              labeled_ratio=args.labeled_ratio, download=True,
                                              transform=datasets.dict_transform['cifar_test'],
                                              unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100
    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    labeled_sampler = BalancedBatchSampler(train_label_set, args.no_class // 2, labeled_batch_size)
    unlabeled_sampler = BalancedBatchSampler(train_unlabel_set, args.no_class,
                                             args.batch_size - labeled_batch_size)
    train_label_loader = torch.utils.data.DataLoader(train_label_set, sampler=labeled_sampler,
                                                     num_workers=0, drop_last=True, batch_size=labeled_batch_size)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, sampler=unlabeled_sampler,
                                                       num_workers=0, drop_last=True,
                                                       batch_size=args.batch_size - labeled_batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

    # First network intialization: pretrain the RotNet network
    model = models.resnet18(num_classes=num_classes)
    model = model.to(device)

    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    discriminator = SimpleDiscriminator()
    discriminator = discriminator.cuda(args.gpu)
    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False
    diffusion_model = Gpt(prototype_sizes=512,
                          predict_xstart=False,
                          max_freq_log2=14,
                          num_frequencies=128,
                          n_embd=512,
                          encoder_depth=1,
                          n_layer=12,
                          n_head=16,
                          len_input=3,
                          attn_pdrop=0.1,
                          resid_pdrop=0.1,
                          embd_pdrop=0.1).cuda(args.gpu)
    diffusion = create_diffusion(learn_sigma=False, predict_xstart=False,
                                 noise_schedule='linear', steps=100)
    timestep_sampler = UniformSampler(diffusion)

    ema_helper = ema.EMAHelper(mu=0.9999)
    ema_helper.register(diffusion_model)
    optimizer_diffusion, lr_scheduler_diffusion = utils_u.make_optimizer(
        diffusion_model.parameters(),
        'sgd', lr=1.e-3, weight_decay=5.e-4, milestones=[30, 50, 80])
    model = model.cuda(args.gpu)
    diffusion_model.cuda(args.gpu)
    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
    optimizer_d = optim.SGD(discriminator.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)
    prototypes = torch.zeros((args.no_class, 512)).cuda(args.gpu)
    alpha = 0.9

    for epoch in tqdm(range(args.epochs)):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer, diffusion,
                           diffusion_model, prototypes)
        prototypes = train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer,
                           diffusion,
                           diffusion_model, optimizer_diffusion, timestep_sampler, prototypes, discriminator,
                           optimizer_d)
        scheduler.step()
        lr_scheduler_diffusion.step()


if __name__ == '__main__':
    main()
