import argparse
import gc
from datetime import datetime
from pathlib import Path
import copy

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
    parser.add_argument("test_path", type=str, default='./testset', help="test set (default:./testset)")
    parser.add_argument("output_result_path", type=str, default='./results/predictions.csv',
                        help="test set (default:'./result/predictions.csv')")

    args = parser.parse_args()
    return args


def main():
    start_time = datetime.now()

    args = parse_args()
    print(args)

    # Compute all of the features of test set using pretrained model.
    print('Loading Model')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model 1
    embed_1, encoder_1 = model_initialization(model_name='efficientnet-b3',
                                              ckpt_path='./all_ckpts/efficientnet-b3_multimask4.pkl', device=device)
    embed_2, encoder_2 = model_initialization(model_name='efficientnet-b2',
                                              ckpt_path='./all_ckpts/efficientnet-b2_multimask4.pkl', device=device)
    embed_3, encoder_3 = model_initialization(model_name='resnet152',
                                              ckpt_path='./all_ckpts/resnet152_multimask2.pkl', device=device)
    embed_4, encoder_4 = model_initialization(model_name='resnet101',
                                              ckpt_path='./all_ckpts/resnet101_multimask3.pkl', device=device)
    print('Done')

    # Load testset
    print('Loading Testset...')
    SIZE = 224
    BATCH_SIZE = 4
    data_transforms = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def collate_fn(data):
        img_names, imgs = list(zip(*data))
        imgs = torch.stack(imgs)
        return img_names, imgs

    dataset = ImageDataset(dataset_path=Path(args.test_path), istrain=False, transforms=data_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                             pin_memory=True, collate_fn=collate_fn)
    print('Done')

    # Acquire testset features.
    print('Getting testset features...')
    list_feat_vector_df = []
    with tqdm(total=len(dataset)) as pbar:
        for i, (img_names, imgs) in enumerate(dataloader):
            pbar.update(BATCH_SIZE)

            with torch.no_grad():
                z1 = get_features(embed_1, encoder_1, device, imgs)
                z2 = get_features(embed_2, encoder_2, device, imgs)
                z3 = get_features(embed_3, encoder_3, device, imgs)
                z4 = get_features(embed_4, encoder_4, device, imgs)

                z = np.concatenate([z1, z2, z3, z4], axis=1)

                gc.disable()  # Disable the garbage collection
                for j, (img_name, z_) in enumerate(zip(img_names, z)):
                    list_feat_vector_df.append(pd.DataFrame({'img_name': img_name, 'z': [z_]}))
                gc.enable()
    feat_vector_df = pd.concat(list_feat_vector_df)
    img_name_query_np = feat_vector_df.img_name.values

    feat_query = np.stack(list(feat_vector_df.z.values), axis=0)  # [num, c]
    feat_query = (feat_query - feat_query.mean(0, keepdims=True)) / feat_query.std(0, keepdims=True)

    # Load Database Features
    print('Load database features...')
    img_name_db_np, z1_np = get_database_features(feature_path='./feature/efficientnet-b3_multimask_last.hdf5',
                                                  model_name='efficientnet-b3')
    img_name_db_np, z2_np = get_database_features(feature_path='./feature/efficientnet-b2_multimask_last.hdf5',
                                                  model_name='efficientnet-b2')
    img_name_db_np, z3_np = get_database_features(feature_path='./feature/resnet152_multimask_last.hdf5',
                                                  model_name='resnet152')
    img_name_db_np, z4_np = get_database_features(feature_path='./feature/resnet101_multimask_last.hdf5',
                                                  model_name='resnet101')

    feat_keys = np.concatenate([z1_np, z2_np, z3_np, z4_np], axis=1)  # [num, c]
    feat_keys = (feat_keys - feat_keys.mean(0, keepdims=True)) / feat_keys.std(0, keepdims=True)


    # QE
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
    TOP_K = 7
    print('Image retrival...')
    index = faiss.IndexFlatL2(feat_keys.shape[1])
    index.add(feat_keys)
    D, I = index.search(feat_query, TOP_K)

    print("Calculate top-7 query results.")
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
    print('Done. Total seconds:{:}s'.format((datetime.now() - start_time).total_seconds()))


def get_database_features(feature_path='./feature/efficientnet-b3_multimask4.hdf5', model_name='efficientnet'):
    f = h5py.File(feature_path, 'r')
    if model_name.startswith('efficientnet'):
        img_name_db_np, z_att_ds, z_att_max_np, z_max_ds = \
            f['img_name_ds'][:], f['z_att_ds'][:], f['z_att_max_ds'][:], f['z_max_ds'][:]
        z_np = np.concatenate([z_att_ds, z_att_max_np, z_max_ds], axis=1)
    elif model_name.startswith('resnet'):
        img_name_db_np, z_att_ds = f['img_name_ds'][:], f['z_att_ds'][:]
        z_np = z_att_ds
    f.close()
    print('Load {:}'.format(feature_path))
    return img_name_db_np, z_np


def get_features(embed, mask_encoder, device, imgs):
    xs = embed(imgs.to(device))  # x_representation:[b, 2048, h, w]
    masks = mask_encoder(xs)  # [b, 1, h, w]
    if embed.model_name.startswith('efficientnet'):
        valid_layer = [2, 3]
        z_ATT = np.concatenate(
            [norm(mask_encoder.attention_pooling(x, mask).squeeze(3).squeeze(2).detach().cpu().numpy()) for
             i, (x, mask) in enumerate(zip(xs, masks)) if i in valid_layer], axis=1)
        z_ATTMAX = np.concatenate([norm(
            F.adaptive_max_pool2d(x * mask, output_size=1).squeeze(3).squeeze(2).detach().cpu().numpy()) for
            i, (x, mask) in enumerate(zip(xs, masks)) if i in valid_layer],
            axis=1)
        z_MAX = norm(F.adaptive_max_pool2d(xs[-1], output_size=1).squeeze(3).squeeze(2).detach().cpu().numpy())
        z = np.concatenate([z_ATT, z_ATTMAX, z_MAX], axis=1)
    elif embed.model_name.startswith('resnet'):
        valid_layer = [3]
        z_ATT = np.concatenate(
            [norm(mask_encoder.attention_pooling(x, mask).squeeze(3).squeeze(2).detach().cpu().numpy()) for
             i, (x, mask) in enumerate(zip(xs, masks)) if i in valid_layer], axis=1)
        z = z_ATT
    return z


def model_initialization(model_name='resnet152', ckpt_path='./ckpts/efficientnet-b3_multimask4.pkl', device='cpu'):
    if model_name.startswith('efficientnet'):
        embed = MultiHeadEffNet(model=model_name, pretrained=False)
    elif model_name.startswith('resnet'):
        embed = MultiHeadResNet(model=model_name, pretrained=False)
    mask_encoder = MultiHeadMaskEncoder(model=embed.model, name=model_name)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        embed.load_state_dict(checkpoint['embed'])
        mask_encoder.load_state_dict(checkpoint['mask_encoder'])

    embed.to(device)
    embed.eval()
    mask_encoder.to(device)
    mask_encoder.eval()

    return embed, mask_encoder


if __name__ == '__main__':
    main()
