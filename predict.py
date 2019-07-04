import argparse
import copy
from pathlib import Path

import faiss
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from model import MultiHeadResNet, MultiHeadEffNet, MultiHeadMaskEncoder
from utils import ImageDataset, norm


def parse_args():
    parser = argparse.ArgumentParser(description="Perform the prediction and write the output to the given path.")
    parser.add_argument("--model-name", type=str, default='resnet152', help="model name (default:resnet152)")
    parser.add_argument("--model-ckpt", type=str, default=None,
                        help="The path of feature set (default:None)")
    parser.add_argument("--feature-path", type=str, default='./feature/resnet152.hdf5',
                        help="feature set (default:./feature/resnet152.hdf5)")
    parser.add_argument("--test-path", type=str, default='./testset', help="test set (default:./testset)")
    parser.add_argument("--output-result-path", type=str, default='./results/resnet152.csv',
                        help="test set (default:'./results/resnet152.csv')")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    # Compute all of the features of test set using pretrained model.
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

    # Load testset
    print('Loading Testset...')
    SIZE = 224
    data_transforms = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def collate_fn(data):
        img_names, imgs = list(zip(*data))
        imgs = torch.stack(imgs)
        return img_names, imgs

    dataset = ImageDataset(dataset_path=Path(args.test_path), istrain=False, transforms=data_transforms)
    BATCH_SIZE = 4
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                             pin_memory=True, collate_fn=collate_fn)
    print('Done')

    # Acquire testset features.
    print('Getting testset features...')
    # Acquire the database features.

    if model_name.startswith('efficientnet'):
        valid_layer = [2, 3]
        img_name_np = []
        vec_length = np.array(embed.out_channels)[valid_layer].sum()
        z_att_np = np.zeros([len(dataset), vec_length], dtype='float32')
        z_att_max_np = np.zeros([len(dataset), vec_length], dtype='float32')
        z_max_np = np.zeros([len(dataset), embed.last_channels], dtype='float32')

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
                    z_MAX = norm(F.adaptive_max_pool2d(xs[-1], output_size=1).squeeze(3).squeeze(2).detach().cpu().numpy())


                for idx, (z_ATT_, z_ATTMAX_, z_MAX_) in enumerate(zip(z_ATT, z_ATTMAX, z_MAX)):
                    z_att_np[i * BATCH_SIZE + idx, :] = z_ATT_
                    z_att_max_np[i * BATCH_SIZE + idx, :] = z_ATTMAX_
                    z_max_np[i * BATCH_SIZE + idx, :] = z_MAX_

        # Save the features
        img_name_query_np = np.array(img_name_np, dtype='object')
        z_np = np.concatenate([z_att_np, z_att_max_np, z_max_np], axis=1)
        feat_query = (z_np - z_np.mean(0, keepdims=True)) / z_np.std(0, keepdims=True)
        print('Done')

        # Load Database Features
        print('Load database features...')
        f = h5py.File(args.feature_path, 'r')
        img_name_db_np, z_att_np, z_att_max_np, z_max_np = \
            f['img_name_ds'][:], f['z_att_ds'][:, -vec_length:], f['z_att_max_ds'][:, -vec_length:], f['z_max_ds'][:]

        z_np = np.concatenate([z_att_np, z_att_max_np, z_max_np], axis=1)

        feat_keys = (z_np - z_np.mean(0, keepdims=True)) / z_np.std(0, keepdims=True)
        print('Done')

    elif model_name.startswith('resnet'):
        valid_layer = [3]
        img_name_np = []
        vec_length = np.array(embed.out_channels)[valid_layer].sum()
        z_att_np = np.zeros([len(dataset), vec_length], dtype='float32')
        # z_att_np = np.zeros([len(dataset), embed.out_channels[-1]], dtype='float32')

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
                    # z_ATT = norm(mask_encoder.attention_pooling(xs[-1], masks[-1]).squeeze(3).squeeze(2).detach().cpu().numpy())

                for idx, (z_ATT_) in enumerate(z_ATT):
                    z_att_np[i * BATCH_SIZE + idx, :] = z_ATT_

        # Save the features
        img_name_query_np = np.array(img_name_np, dtype='object')
        z_np = z_att_np
        feat_query = (z_np - z_np.mean(0, keepdims=True)) / z_np.std(0, keepdims=True)
        print('Done')

        # Load Database Features
        print('Load database features...')
        f = h5py.File(args.feature_path, 'r')
        img_name_db_np, z_att_np = f['img_name_ds'][:], f['z_att_ds'][:, -vec_length:]
        # img_name_db_np, z_att_ds = f['img_name_ds'][:], f['z_att_ds'][:, -embed.out_channels[-1]:]
        z_np = z_att_np
        feat_keys = (z_np - z_np.mean(0, keepdims=True)) / z_np.std(0, keepdims=True)
        print('Done')


    # DBA
    # print('DBA')
    # DBA_num = 2
    # index_flat = faiss.IndexFlatL2(feat_keys.shape[1])
    # index_flat.add(feat_keys)
    # D, I = index_flat.search(feat_keys, DBA_num)
    #
    # new_feat_keys = copy.deepcopy(feat_keys)
    # for num in range(len(I)):
    #     new_feat = feat_keys[I[num][0]]
    #     for num1 in range(1, len(I[num])):
    #         weight = (len(I[num]) - num1) / float(len(I[num]))
    #         new_feat += feat_keys[num1] * weight
    #     new_feat_keys[num] = new_feat
    # print('Done')

    #QE
    # print('Query Expansion')
    # QE_num = 2
    # index_flat = faiss.IndexFlatL2(feat_keys.shape[1])
    # index_flat.add(feat_keys)
    # D, I = index_flat.search(feat_query, QE_num - 1)
    # new_feat_query=copy.deepcopy(feat_query)
    # for num in range(len(new_feat_query)):
    #     new_feat_query[num] = (new_feat_query[num] + feat_keys[I[num][0]]) / float(QE_num)
    # print('Done')

    # Image Retrival
    print('Image retrival...l2')
    index = faiss.IndexFlatL2(feat_keys.shape[1])
    index.add(feat_keys)

    TOP_K = 7
    D, I = index.search(feat_query, TOP_K)

    # Save query results
    query_res = []
    for i, topk_idx in enumerate(I):
        img_query_id = img_name_query_np[i].split('.')[0]

        query_sample_res = [img_query_id]
        for j, jth_idx in enumerate(topk_idx):
            img_key_id = img_name_db_np[jth_idx].split('.')[0]
            query_sample_res.append(img_key_id)
        query_res.append(query_sample_res)

    query_res = pd.DataFrame(query_res)
    query_res.rename(columns={0: 'Query'}, inplace=True)
    query_res.to_csv(args.output_result_path, index=False, header=False)
    print('Done')


if __name__ == '__main__':
    main()
