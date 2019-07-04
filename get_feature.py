import argparse
import gc
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from model import MultiHeadResNet, MultiHeadEffNet, MultiHeadMaskEncoder
from utils import ImageDataset, norm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract global features from image to form a feature database.")
    parser.add_argument("--model-name", type=str, default='resnet152', help="model name (default:resnet152)")
    parser.add_argument("--dataset", type=str, default='/data/wangjiawei/dataset/AIMeetsBeauty',
                        help='The path of training set (default: /data/wangjiawei/dataset/AIMeetsBeauty)')
    parser.add_argument("--model-ckpt", type=str, default='./ckpts/resnet152.pkl',
                        help="The path of feature set (default:./ckpts/resnet152.pkl")
    parser.add_argument("--output-feature-path", type=str, default='./feature/resnet152.hdf5',
                        help="The path of feature set (default:./feature/resnet152.hdf5)")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # Load models
    print('Loading Model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = args.model_name
    if model_name.startswith('efficientnet'):
        embed = MultiHeadEffNet(model=model_name, pretrained=False)
    elif model_name.startswith('resnet'):
        embed = MultiHeadResNet(model=model_name, pretrained=False)

    mask_encoder = MultiHeadMaskEncoder(model=embed.model, name=model_name)
    if args.model_ckpt:
        checkpoint = torch.load(args.model_ckpt, map_location='cpu')
        embed.load_state_dict(checkpoint['embed'])
        mask_encoder.load_state_dict(checkpoint['mask_encoder'])
        print('Load state dict.')

    embed.to(device)
    embed.eval()
    mask_encoder.to(device)
    mask_encoder.eval()
    print('Done')

    # Load dataset
    print('Loading dataset...')
    SIZE = 224
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])


    def collate_fn(data):
        img_names, imgs = list(zip(*data))
        imgs = torch.stack(imgs)
        return img_names, imgs


    BATCH_SIZE = 4
    dataset = ImageDataset(dataset_path=Path(args.dataset), size=SIZE, istrain=True, transforms=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6,
                                             pin_memory=True, collate_fn=collate_fn)
    print('Done')

    # Acquire the database features.
    if model_name.startswith('efficientnet'):
        valid_layer = [2, 3]
        vec_length = np.array(embed.out_channels)[valid_layer].sum()
        img_name_np = []
        z_att_np = np.zeros([len(dataset), vec_length], dtype='float32')
        z_att_max_np = np.zeros([len(dataset), vec_length], dtype='float32')
        z_max_np = np.zeros([len(dataset), embed.last_channels], dtype='float32')
        gc.disable()
        with tqdm(total=len(dataset)) as pbar:
            for i, (img_names, imgs) in enumerate(dataloader):
                pbar.update(BATCH_SIZE)

                with torch.no_grad():
                    img_name_np.extend(img_names)
                    xs = embed(imgs.to(device))  # x_representation:[b, 2048, h, w]
                    masks = mask_encoder(xs)  # [b, 1, h, w]
                    z_ATT = np.concatenate(
                        [norm(mask_encoder.attention_pooling(x, mask).squeeze(3).squeeze(2).detach().cpu().numpy()) for
                         i, (x, mask) in enumerate(zip(xs, masks)) if i in valid_layer], axis=1)
                    z_ATTMAX = np.concatenate([norm(
                        F.adaptive_max_pool2d(x * mask, output_size=1).squeeze(3).squeeze(2).detach().cpu().numpy()) for
                        i, (x, mask) in enumerate(zip(xs, masks)) if i in valid_layer],
                        axis=1)
                    z_MAX = norm(
                        F.adaptive_max_pool2d(xs[-1], output_size=1).squeeze(3).squeeze(2).detach().cpu().numpy())

                    for idx, (z_ATT_, z_ATTMAX_, z_MAX_) in enumerate(zip(z_ATT, z_ATTMAX, z_MAX)):
                        z_att_np[i * BATCH_SIZE + idx, :] = z_ATT_
                        z_att_max_np[i * BATCH_SIZE + idx, :] = z_ATTMAX_
                        z_max_np[i * BATCH_SIZE + idx, :] = z_MAX_

        gc.enable()

        # Save the features
        img_name_np = np.array(img_name_np, dtype='object')
        with h5py.File(args.output_feature_path, 'w') as f:
            f.create_dataset('img_name_ds', shape=img_name_np.shape, data=img_name_np,
                             dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('z_att_ds', shape=z_att_np.shape, data=z_att_np)
            f.create_dataset('z_att_max_ds', shape=z_att_max_np.shape, data=z_att_max_np)
            f.create_dataset('z_max_ds', shape=z_max_np.shape, data=z_max_np)

    elif model_name.startswith('resnet'):
        valid_layer = [3]
        vec_length = np.array(embed.out_channels)[valid_layer].sum()
        img_name_np = []
        z_att_np = np.zeros([len(dataset), vec_length], dtype='float32')
        gc.disable()
        with tqdm(total=len(dataset)) as pbar:
            for i, (img_names, imgs) in enumerate(dataloader):
                pbar.update(BATCH_SIZE)

                with torch.no_grad():
                    img_name_np.extend(img_names)
                    xs = embed(imgs.to(device))  # x_representation:[b, 2048, h, w]
                    masks = mask_encoder(xs)  # [b, 1, h, w]
                    z_ATT = np.concatenate(
                        [norm(mask_encoder.attention_pooling(x, mask).squeeze(3).squeeze(2).detach().cpu().numpy()) for
                         i, (x, mask) in enumerate(zip(xs, masks)) if i in valid_layer], axis=1)

                    for idx, (z_ATT_) in enumerate(z_ATT):
                        z_att_np[i * BATCH_SIZE + idx, :] = z_ATT_

        gc.enable()

        # Save the features
        img_name_np = np.array(img_name_np, dtype='object')
        with h5py.File(args.output_feature_path, 'w') as f:
            f.create_dataset('img_name_ds', shape=img_name_np.shape, data=img_name_np,
                             dtype=h5py.special_dtype(vlen=str))
            f.create_dataset('z_att_ds', shape=z_att_np.shape, data=z_att_np)
