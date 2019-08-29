import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from config import device, print_freq, grad_clip
from data_gen import VoxCeleb1Dataset, pad_collate
from models.arc_margin import ArcMarginModel
from models.models import GST
from utils import parse_args, save_checkpoint, AverageMeter, get_logger, accuracy, clip_gradient


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        model = GST()
        metric_fc = ArcMarginModel(args)

        print(model)
        # model = nn.DataParallel(model)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('total params: ' + str(total_params))
        print('trainable params: ' + str(trainable_params))

        # optimizer
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=args.lr,
                                     betas=(0.9, 0.98), eps=1e-09)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        metric_fc = checkpoint['metric_fc']
        optimizer = checkpoint['optimizer']
        optimizer.update_lr(args.lr)

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_dataset = VoxCeleb1Dataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=True, num_workers=args.num_workers)
    valid_dataset = VoxCeleb1Dataset(args, 'valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
                                               pin_memory=True, shuffle=False, num_workers=args.num_workers)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)
        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_accuracy', train_acc, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model/learning_rate', lr, epoch)
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        valid_loss, valid_acc = valid(valid_loader=valid_loader,
                                      model=model,
                                      metric_fc=metric_fc,
                                      criterion=criterion,
                                      logger=logger)
        writer.add_scalar('model/valid_loss', valid_loss, epoch)
        writer.add_scalar('model/valid_accuracy', valid_acc, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, metric_fc, optimizer, best_loss, is_best)


def train(train_loader, model, metric_fc, criterion, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, input_lengths, label = data
        padded_input = padded_input.to(device)
        # input_lengths = input_lengths.to(device)
        label = label.to(device)

        # Forward prop.
        feature = model(padded_input)  # embedding => [N, 512]
        output = metric_fc(feature, label)  # class_id_out => [N, 1251]

        # Calculate loss
        loss = criterion(output, label)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        top1_accuracy = accuracy(output, label, 1)
        accs.update(top1_accuracy)

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                        'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(epoch, i, len(train_loader), loss=losses,
                                                                          accs=accs))

    return losses.avg, accs.avg


def valid(valid_loader, model, metric_fc, criterion, logger):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, input_lengths, label = data
        padded_input = padded_input.to(device)
        # input_lengths = input_lengths.to(device)
        label = label.to(device)

        # Forward prop.
        with torch.no_grad():
            feature = model(padded_input)  # embedding => [N, 512]
            output = metric_fc(feature, label)  # class_id_out => [N, 1251]

        # Calculate loss
        loss = criterion(output, label)

        # Keep track of metrics
        losses.update(loss.item())
        top1_accuracy = accuracy(output, label, 1)
        accs.update(top1_accuracy)

    # Print status
    logger.info('\nValidation Loss {loss.avg:.5f}\tAccuracy {accs.avg:.3f}\n'.format(loss=losses, accs=accs))

    return losses.avg, accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
