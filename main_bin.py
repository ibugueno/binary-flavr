import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
import myutils
from loss import Loss
from model.binFLAVR_arch import UNet_3D_3D
from util import BinOp


# === Helper: Checkpoint loader ===
def load_checkpoint(args, model, optimizer, path):
    print("Loading checkpoint:", path)
    checkpoint = torch.load(path)
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint.get("lr", args.lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# === Setup ===
args, unparsed = config.get_args()
print(args)

device = torch.device('cuda' if args.cuda else 'cpu')
torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# === Output folders ===
save_loc = os.path.join(args.checkpoint_dir, "saved_models_final", args.dataset, args.exp_name)
os.makedirs(save_loc, exist_ok=True)

with open(os.path.join(save_loc, "opts.txt"), "w") as fh:
    fh.write(str(args))

writer_loc = os.path.join(args.checkpoint_dir, f'tensorboard_logs_{args.dataset}_final', args.exp_name)
writer = SummaryWriter(writer_loc)


# === Load Dataset ===
if args.dataset == "vimeo90K_septuplet":
    from dataset.vimeo90k_septuplet import get_loader
    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)
elif args.dataset == "gopro":
    from dataset.GoPro import get_loader
    train_loader = get_loader(args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers, test_mode=False, interFrames=args.n_outputs, n_inputs=args.nbr_frame)
    test_loader = get_loader(args.data_root, args.batch_size, shuffle=False, num_workers=args.num_workers, test_mode=True, interFrames=args.n_outputs, n_inputs=args.nbr_frame)
else:
    raise NotImplementedError


# === Build Model ===
print("Building model:", args.model.lower())
model = UNet_3D_3D(args.model.lower(), n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)
model = torch.nn.DataParallel(model).to(device)

# === Binarization handler ===
bin_op = BinOp(model)

# === Loss and Optimizer ===
criterion = Loss(args)
optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


# === Training ===
def train(args, epoch):
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.train()
    criterion.train()

    for i, (images, gt_image) in enumerate(train_loader):
        images = [img.cuda() for img in images]
        gt = [g.cuda() for g in gt_image]

        optimizer.zero_grad()
        bin_op.binarization()

        with torch.autograd.set_detect_anomaly(True):
            out = model(images)
            out = torch.cat(out)
            gt = torch.cat(gt)

            loss, loss_specific = criterion(out, gt)
            loss.backward()


        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()

        for k, v in losses.items():
            if k != 'total':
                v.update(loss_specific[k].item())
        losses['total'].update(loss.item())

        if i % args.log_iter == 0:
            myutils.eval_metrics(out, gt, psnrs, ssims)
            print(f'Train Epoch: {epoch} [{i}/{len(train_loader)}]\tLoss: {losses["total"].avg:.6f}\tPSNR: {psnrs.avg:.4f}')

            timestep = epoch * len(train_loader) + i
            writer.add_scalar('Loss/train', loss.item(), timestep)
            writer.add_scalar('PSNR/train', psnrs.avg, timestep)
            writer.add_scalar('SSIM/train', ssims.avg, timestep)
            writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], timestep)

            losses, psnrs, ssims = myutils.init_meters(args.loss)


# === Evaluation ===
def test(args, epoch):
    print(f"Evaluating for epoch = {epoch}")
    losses, psnrs, ssims = myutils.init_meters(args.loss)
    model.eval()
    criterion.eval()

    with torch.no_grad():
        for i, (images, gt_image) in enumerate(tqdm(test_loader)):
            images = [img.cuda() for img in images]
            gt = [g.cuda() for g in gt_image]

            out = model(images)
            out = torch.cat(out)
            gt = torch.cat(gt)

            loss, loss_specific = criterion(out, gt)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            myutils.eval_metrics(out, gt, psnrs, ssims)

    print(f"Loss: {losses['total'].avg:.4f}, PSNR: {psnrs.avg:.4f}, SSIM: {ssims.avg:.4f}\n")

    with open(os.path.join(save_loc, 'results.txt'), 'a') as f:
        f.write(f'For epoch={epoch}\tPSNR: {psnrs.avg:.4f}, SSIM: {ssims.avg:.4f}\n')

    timestep = epoch + 1
    writer.add_scalar('Loss/test', loss.item(), timestep)
    writer.add_scalar('PSNR/test', psnrs.avg, timestep)
    writer.add_scalar('SSIM/test', ssims.avg, timestep)

    return losses['total'].avg, psnrs.avg, ssims.avg


# === Main Loop ===
def main(args):
    if args.pretrained:
        state_dict = torch.load(args.pretrained)['state_dict']
        model_dict = model.state_dict()
        for k, v in state_dict.items():
            if v.shape == model_dict[k].shape:
                print("Loading", k)
                model_dict[k] = v
            else:
                print("Not loading", k)
        model.load_state_dict(model_dict)

    best_psnr = 0
    for epoch in range(args.start_epoch, args.max_epoch):
        train(args, epoch)
        test_loss, psnr, _ = test(args, epoch)

        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        myutils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_psnr,
            'lr': optimizer.param_groups[-1]['lr']
        }, save_loc, is_best, args.exp_name)

        scheduler.step(test_loss)


if __name__ == "__main__":
    main(args)
