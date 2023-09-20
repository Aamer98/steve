import math
import os.path
import argparse

import torch
import torchvision.utils as vutils

from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from steve import STEVE
from data import GlobVideoDataset
from utils import cosine_anneal, linear_warmup

#models innit
from utils import *
from dvae import dVAE
from transformer import TransformerEncoder, TransformerDecoder


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--ep_len', type=int, default=3)

parser.add_argument('--checkpoint_path', type=str, default='/home/as26840@ens.ad.etsmtl.ca/repos/steve/logs/hmdb/checkpoint.pt.tar')
parser.add_argument('--data_path', type=str, default='/home/as26840@ens.ad.etsmtl.ca/data/hmdb51_steve/*')
parser.add_argument('--log_path', type=str, default='/home/as26840@ens.ad.etsmtl.ca/repos/steve/logs/hmdb/')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--steps', type=int, default=200000)

parser.add_argument('--num_iterations', type=int, default=2)
parser.add_argument('--num_slots', type=int, default=15)
parser.add_argument('--cnn_hidden_size', type=int, default=64)
parser.add_argument('--slot_size', type=int, default=192)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_predictor_blocks', type=int, default=1)
parser.add_argument('--num_predictor_heads', type=int, default=4)
parser.add_argument('--predictor_dropout', type=int, default=0.0)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_blocks', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--hard', action='store_true')
parser.add_argument('--use_dp', default=False, action='store_true')

parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text('hparams', arg_str)

train_dataset   = GlobVideoDataset(root=args.data_path, phase='train', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')
val_dataset     = GlobVideoDataset(root=args.data_path, phase='val', img_size=args.image_size, ep_len=args.ep_len, img_glob='????????_image.png')

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

train_loader   = DataLoader(train_dataset, sampler=None, **loader_kwargs)
val_loader     = DataLoader(val_dataset, sampler=None, **loader_kwargs)

train_epoch_size = len(train_loader)
val_epoch_size   = len(val_loader)

log_interval = train_epoch_size // 5

model = STEVE(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    model.load_state_dict(checkpoint['model'])
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

model = model.to(device)
if args.use_dp:
    model = DP(model)

optimizer = Adam([
    {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
    {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
    {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
])

if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

# to visualize vq-vae reconstruction
def visualize(video, recon_dvae, recon_tf, attns, N=8):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        recon_tf_t = recon_tf[:N, t, None, :, :, :]
        attns_t = attns[:N, t, :, :, :, :]

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, recon_tf_t, attns_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(args.num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


# Prototyping
epoch = 0
batch = 0
video = next(iter(train_loader))

model.train()
global_step = epoch * train_epoch_size + batch

tau = cosine_anneal(
    global_step,
    args.tau_start,
    args.tau_final,
    0,
    args.tau_steps)

lr_warmup_factor_enc = linear_warmup(
    global_step,
    0.,
    1.0,
    0.,
    args.lr_warmup_steps)

lr_warmup_factor_dec = linear_warmup(
    global_step,
    0.,
    1.0,
    0,
    args.lr_warmup_steps)

lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

optimizer.param_groups[0]['lr'] = args.lr_dvae
optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

video = video.to(device)

optimizer.zero_grad()


# init networks


B, T, C, H, W = video.size()

#for dvae, since no time dimension #TODO: How is this reconstructed along time for temporal info
video_flat = video.flatten(end_dim=1)                               # B * T, C, H, W




# (recon, cross_entropy, mse, attns) = model(video, tau, args.hard)

# loss = mse + cross_entropy

# loss.backward()
# clip_grad_norm_(model.parameters(), args.clip, 'inf')
# optimizer.step()
breakpoint()