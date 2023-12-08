import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import models
import open_world_cifar as datasets
import utils_u
from diffusion_u import create_diffusion, ema, timestep_sampler
from diffusion_u.timestep_sampler import UniformSampler
from models.transformer import Gpt
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer, diffusion,
          diffusion_model, diffusion_optimizer, timestep_sampler):
    model.train()
    diffusion_model.train()

    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        ((ux, ux2), _) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        ux, ux2 = ux.to(device), ux2.to(device)
        optimizer.zero_grad()
        diffusion_optimizer.zero_grad()
        output, feat = model(x)
        with torch.no_grad():
            logits_x_ulb_w, feat2 = model(ux2)

        labeled_len = len(target)
        # Compute context vectors for unlabeled data

        pseudo_labels = F.softmax(logits_x_ulb_w, dim=1)
        y_c_u = compute_context_vector(feat[labeled_len:, ], logits_x_ulb_w, 5, 0.9)
        logits_x_lb = output[:labeled_len, :]
        y_c_l = compute_context_vector(feat[:labeled_len, :], logits_x_lb, 5, 0.9)

        t, vlb_weights = timestep_sampler.sample(labeled_len, args.gpu)
        t = t.cuda(args.gpu)

        # diffusion_loss, denoised_labeld = diffusion.training_losses(diffusion_model,
        #                                                             logits_x_ulb_w, y_c_u, t)

        diffusion_loss_, denoised_labeld_ = diffusion.training_losses(diffusion_model,
                                                                      logits_x_lb, y_c_l, t)
        p_sample = diffusion.p_sample(diffusion_model, logits_x_ulb_w, y_c_u,
                                      torch.ones(logits_x_ulb_w.shape[0], dtype=torch.long).cuda(args.gpu))
        p_sample_l = diffusion.p_sample(diffusion_model, logits_x_lb, y_c_l,
                                        torch.ones(logits_x_lb.shape[0], dtype=torch.long).cuda(args.gpu))
        # Compute total loss and backpropagate
        clean_logits_u = p_sample["pred_xstart"]
        labels_denoised_labeld = F.softmax(clean_logits_u, dim=1)
        clean_logits_l = p_sample_l["pred_xstart"]
        unsup_loss = compute_unsup_loss(labels_denoised_labeld, clean_logits_u, args.gpu, num_classes=args.no_class,
                                        unseen_classes=[])
        ce_loss = F.cross_entropy(clean_logits_l, target)
        prob = F.softmax(torch.cat((clean_logits_l, clean_logits_u), dim=0), dim=1)
        entropy_loss = entropy(torch.mean(prob, 0))

        loss = - entropy_loss + ce_loss + diffusion_loss_.mean() + unsup_loss

        # bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        diffusion_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        diffusion_optimizer.step()

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)


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


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer, diffusion, diffusion_model):
    global best_acc, best_seen_acc, best_unseen_acc
    model.eval()
    diffusion_model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            t = torch.ones(output.shape[0], dtype=torch.long).cuda(args.gpu)
            t = t.cuda(args.gpu)
            denoised_labeld = diffusion.p_sample(diffusion_model,
                                                 output, F.softmax(output, dim=1), t)

            output = denoised_labeld["pred_xstart"]
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


def compute_unsup_loss(denoised_soft_labels, logits_x_ulb_s, gpu, num_classes=10, unseen_classes=[]):
    # Convert logits to probabilities
    probs = F.softmax(logits_x_ulb_s, dim=1)

    # Find the pseudo-labels and their confidence
    pseudo_labels = torch.argmax(denoised_soft_labels, dim=1)
    max_probs, _ = torch.max(probs, dim=1)

    # Select samples where the confidence is above the threshold
    confident_samples = max_probs > 0.0
    confident_labels = pseudo_labels[confident_samples]

    # Balance the set
    unique_labels, counts = confident_labels.unique(return_counts=True)
    if len(unique_labels) < num_classes:
        return torch.tensor(0.0).cuda(gpu)

    min_samples = counts.min()

    balanced_indices = torch.cat(
        [confident_samples.nonzero()[confident_labels == label][:min_samples] for label in unique_labels])

    # Compute loss only on the balanced subset
    balanced_logits = logits_x_ulb_s[balanced_indices.squeeze(1)]
    balanced_labels = pseudo_labels[balanced_indices.squeeze(1)]
    # balanced_labels = torch.where(balanced_labels < 5, balanced_labels + 5, balanced_labels)

    # Assuming a standard cross-entropy loss for simplicity
    loss = F.cross_entropy(balanced_logits, balanced_labels)
    return loss


def compute_context_vector(x_u, pseudo_labels, k, tau):
    # Assuming x is of shape (batch_size, num_features)
    # and pseudo_labels is of shape (batch_size, num_classes)
    batch_size, num_classes = pseudo_labels.shape

    # Normalize x to have unit norm, which is required for cosine similarity
    x_normalized = F.normalize(x_u, p=2, dim=1)

    # Compute cosine similarity, shape: (batch_size, batch_size)
    similarities = torch.mm(x_normalized, x_normalized.t())

    # Get indices of k nearest neighbors (excluding self), shape: (batch_size, k)
    _, neighbors_indices = torch.topk(similarities, k + 1, largest=True, sorted=False)
    neighbors_indices = neighbors_indices[:, 1:]  # Exclude self

    # Gather k nearest pseudo labels, shape: (batch_size, k, num_classes)
    neighbors_pseudo_labels = pseudo_labels[neighbors_indices.reshape(-1)].reshape(-1, k, num_classes)

    # Compute weights using cosine similarity, shape: (batch_size, k)
    similarities = similarities.gather(1, neighbors_indices)
    weights = F.softmax(similarities / tau, dim=1)

    # Expand dimensions for proper matrix multiplication
    weights = weights.unsqueeze(2)  # shape: (batch_size, k, 1)

    # Compute context vectors, shape: (batch_size, num_classes)
    y_c = torch.sum(weights * neighbors_pseudo_labels, dim=1)
    # y_c=F.softmax(y_c, dim=1)

    return y_c


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
    device = torch.device("cuda" if args.cuda else "cpu")

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
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,
                                                       batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    # First network intialization: pretrain the RotNet network
    model = models.resnet18(num_classes=num_classes)
    model = model.to(device)

    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False
    diffusion_model = Gpt(prototype_sizes=args.no_class,
                          predict_xstart=False,
                          max_freq_log2=14,
                          num_frequencies=128,
                          n_embd=args.no_class,
                          encoder_depth=1,
                          n_layer=1,
                          n_head=1,
                          len_input=3,
                          attn_pdrop=0.0,
                          resid_pdrop=0.0,
                          embd_pdrop=0.0).cuda(args.gpu)
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
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in tqdm(range(args.epochs)):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer, diffusion,
                           diffusion_model)
        train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer, diffusion,
              diffusion_model, optimizer_diffusion, timestep_sampler)
        scheduler.step()
        lr_scheduler_diffusion.step()


if __name__ == '__main__':
    main()
