import argparse
import datetime
import warnings
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')

from model import MultiHeadResNet, MultiHeadEffNet, MultiHeadMaskEncoder
from utils import PairedTransformImageWithMaskDataset, denormalize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training an encoder to extract global features from image using self supervised learning')
    parser.add_argument("--model-name", type=str, default='resnet152', help="model name (default:resnet152)")
    parser.add_argument("--dataset", type=str, default='/data/wangjiawei/dataset/AIMeetsBeauty',
                        help='The path of training set (default: /data/wangjiawei/dataset/AIMeetsBeauty)')
    parser.add_argument("--epochs", type=int, default=30, help='epochs (default:30)')
    parser.add_argument("--batch-size", type=int, default=64, help='batch size (default:64)')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate (default:0.0001)')
    args = parser.parse_args()
    return args


def train(dataloader, optimizer):
    global iteration, ema, beta

    with tqdm(total=len(dataset)) as pbar:
        for i, (img_names, imgs, augmented_img, matrix, img_mask, img_fg_mask) in enumerate(dataloader):
            pbar.update(args.batch_size)
            iteration += 1

            imgs, augmented_img, matrix, img_mask, img_fg_mask = imgs.to(device), augmented_img.to(device), matrix.to(
                device), img_mask.to(device), img_fg_mask.to(device)

            with torch.no_grad():
                xs = embed(torch.cat([imgs, augmented_img], dim=0))  # x_representation:[b, 2048, h, w]

            masks = mask_encoder(xs)  # [b, 1, h, w]

            network_name = ['mask_encoder']
            for n in network_name:
                optimizer[n].zero_grad()

            loss_mse = 0
            for mask in masks:
                loss_mse += F.mse_loss(mask, F.interpolate(torch.cat([img_mask, img_fg_mask], dim=0),
                                                     (mask.size(2), mask.size(3))))

            loss_mse.backward()
            for n in network_name:
                optimizer[n].step()

            ema = 0.99 * ema + 0.01 * loss_mse.item()


            writer.add_scalar('loss/loss_mse', loss_mse.item(), iteration)

            if iteration % 1024 == 0:
                writer.add_images('img/original', denormalize(imgs[:4].cpu()), iteration)
                writer.add_images('img/augmented', denormalize(augmented_img[:4].cpu()), iteration)
                writer.add_images('mask/original', img_mask[:4].cpu().expand(4, 3, img_mask.size(2), img_mask.size(3)),
                                  iteration)
                writer.add_images('mask/img_fg',
                                  img_fg_mask[:4].cpu().expand(4, 3, img_mask.size(2), img_mask.size(3)), iteration)

                for i, mask in enumerate(masks):
                    writer.add_images('img_mask_predict/{:}block'.format(i), mask[:4].cpu().expand(4, 3, mask.size(2), mask.size(3)), iteration)
                    writer.add_images('img_fg_mask_predict/{:}block'.format(i), mask[args.batch_size:args.batch_size+4].cpu().expand(4, 3, mask.size(2), mask.size(3)), iteration)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.manual_seed(2019)
    random.seed(2019)

    filename = datetime.datetime.strftime(datetime.datetime.now(),
                                          '%Y-%m-%d-%H-%M-%S-name{:}-bs{:}'.format(args.model_name, args.batch_size))
    writer = SummaryWriter(log_dir='./runs/' + filename)

    # Load dataset
    print('Loading dataset')


    def collate_fn(data):
        img_names, imgs, augmented_img, matrix, img_mask, img_fg_mask = list(zip(*data))
        imgs = torch.stack(imgs)
        augmented_img = torch.stack(augmented_img)
        matrix = torch.from_numpy(np.stack(matrix))
        img_mask = torch.stack(img_mask)
        img_fg_mask = torch.stack(img_fg_mask)
        return (img_names, imgs, augmented_img, matrix, img_mask, img_fg_mask)


    SIZE = 224
    PATCH = args.batch_size

    dataset = PairedTransformImageWithMaskDataset(dataset_path=Path(args.dataset),
                                                  size=SIZE,
                                                  data_transforms=transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                                                                      transforms.ToTensor(),
                                                                                      transforms.Normalize(
                                                                                          mean=(0.485, 0.456, 0.406),
                                                                                          std=(0.229, 0.224, 0.225))]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             pin_memory=True, collate_fn=collate_fn, drop_last=True)
    print('Done')

    # Load models
    print('Loading Model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_name
    if model_name.startswith('efficientnet'):
        embed = MultiHeadEffNet(model=model_name, pretrained=True)
    elif model_name.startswith('resnet'):
        embed = MultiHeadResNet(model=model_name, pretrained=True)

    mask_encoder = MultiHeadMaskEncoder(model=embed.model, name=model_name)

    embed.to(device)
    mask_encoder.to(device)

    embed.eval()
    mask_encoder.train()
    print('Done')

    optimizer = {'embed': optim.Adam(embed.parameters(), lr=args.lr),
                 'mask_encoder': optim.Adam(mask_encoder.parameters(), lr=args.lr),
                 }

    iteration = 0
    ema = 1
    best_ema = 1e2
    beta = 0
    for epoch in range(args.epochs):
        print('Epoch:{:} Running...'.format(epoch))
        train(dataloader=dataloader, optimizer=optimizer)

        if ema < best_ema:
            best_ema = ema
            state = {'embed': embed.state_dict(),
                     'mask_encoder': mask_encoder.state_dict(),
                     }
            torch.save(state, './ckpts/{:}'.format(args.model_name + '_multimask.pkl'))
            print('save model{:}, best_ema{:.4f}'.format(epoch, best_ema))
