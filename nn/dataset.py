# https://www.drivendata.co/blog/hateful-memes-benchmark/ Good place to start
import math
import os
from collections import namedtuple
from PIL.Image import new
from numpy.core.fromnumeric import size
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from math import floor
import random
import numpy as np
import pickle
from tqdm import tqdm
from util.misc import bcolors
import csv
from util.range_image import pcd_file_to_range_image, range_projection
from util.transforms import build_se3_transform
import re


def preprocess(image, rf, qx, qy, tf, HAS_LABEL):
    '''
    Expects image input to be torch tensor. 0-255
    '''
    assert image.shape == torch.Size([2, 16, 1800]), 'Wrong image shape!'

    # Label represents percentage increase in features
    # label = ((tf - rf) / rf) * 100 if HAS_LABEL else 0
    # label = ((tf - rf) / 50) if HAS_LABEL else 0
    label = ((tf - rf) / rf) if HAS_LABEL else 0
    # label = np.tanh(label**4 * 1000)
    # label = tf / 500 if HAS_LABEL else 0
    # TODO: Proper nomralization
    data = {
        'im_inp': image.float() / 45.0,
        'sc_inp': torch.tensor((qx / 10.0, qy / 10.0, rf/50.0)).float(),
        'y': torch.tensor([label]).float()
    }
    return data


class LidarDataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.range_img_dir = self.data_dir + 'range_images/'
        self.count_img_dir = self.data_dir + 'count_images/'
        self.pickle_dir = self.data_dir + 'pickles/'
        self.pcd_dir = self.data_dir + 'pointclouds/'
        self.filenames = self.data_dir + 'filenames.csv'
        self.len = len(os.listdir(self.pickle_dir))
        # self.len = 100
        self.data_list = self._create_data_list()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_list_entry = self.data_list[idx]

        # image = read_image(
        #     self.data_list[idx].range_image_file, ImageReadMode.GRAY)
        range_image = torch.tensor(
            np.array(np.load(self.data_list[idx].range_image_file)))
        count_image = torch.tensor(
            np.array(np.load(self.data_list[idx].count_image_file)))
        image = torch.stack([range_image, count_image])
        # image = torch.unsqueeze(image, 0)
        data = preprocess(image, data_list_entry.rf, data_list_entry.qx,
                          data_list_entry.qy, data_list_entry.tf, HAS_LABEL=True)
        return data

    def _create_data_list(self):
        """Create a datalist from self.data_dir"""
        data_list = []
        DataListEntry = namedtuple(
            'DataListEntry', ['range_image_file', 'count_image_file', 'rf', 'qx', 'qy', 'tf'])

        with open(self.filenames, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # pcd, range image, count image, pickle
            for itr, row in enumerate(tqdm(reader, total=self.len, desc="DataLoader")):
                if itr == 0:
                    continue
                pcd = np.array(np.load(self.pcd_dir + row[0]))
                range_image_file = self.range_img_dir + row[1]
                count_image_file = self.range_img_dir + row[2]
                pickle_file = self.pickle_dir + row[3]
                with open(pickle_file, 'rb') as f:
                    pkl = pickle.load(f)
                data_list.append(
                    DataListEntry(range_image_file, count_image_file, len(pcd),
                                  pkl[0], pkl[1], pkl[2])
                )
        return data_list

    def preprocess_old(self, dataentry):
        """Preprocess data before training and testing"""

        im_inp = dataentry['count_image'].float() / 255.0
        sc_inp = torch.vstack(
            [dataentry['qx'] / 10.0, dataentry['qy'] / 10.0, dataentry['rf']/300.0]).T.float()
        y = (dataentry['tf'] - dataentry['rf']) / dataentry['rf']
        y = y.reshape((len(y), -1)).float()
        return im_inp, sc_inp, y

    def get_split_sizes(self, train_percent, test_percent):
        """Handle size mismatches due to percentage multiplication"""

        train_size, test_size = [
            int(self.len * train_percent), int(self.len * test_percent)]
        if train_size + test_size > self.len:
            train_size -= train_size - (self.len - test_size)
        elif train_size + test_size < self.len:
            test_size += (self.len - train_size) - test_size
        return train_size, test_size


class DriftDataset(Dataset):
    def __init__(self, filenames) -> None:
        super().__init__()
        self.filenames = filenames
        self.data_list = self._create_data_list()
        self.len = len(self.data_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pcdname, driftname = self.data_list[idx]
        match1 = re.search(r'/\d/', pcdname)
        match2 = re.search(r'-\d.npy', pcdname)

        pcdlname = pcdname[:match1.start()] + '/0/' + pcdname[match1.end():match2.start()] + '-0.npy' 
        pcdrname = pcdname[:match1.start()] + '/4/' + pcdname[match1.end():match2.start()]  +'-4.npy'
        pcdcname = pcdname

        pcdl = np.load(pcdlname)
        pcdr = np.load(pcdrname)
        pcdc = np.load(pcdcname)

        drift = np.load(driftname)

        # if np.random.random() < 0.5:
        #     tf = build_se3_transform([0, 0, 0, 0, 0, np.pi])
        #     pts = pcd[:, :3].T
        #     zeros = np.zeros(pts.shape[1]).reshape(1, -1)
        #     pts = np.vstack([pts, zeros])
        #     pts = tf @ pts

        #     pcd[:, :3] = pts.T[:, :3]
        #     drift[0], drift[-1] = drift[-1], drift[0]
        rangel, _, _, _ = range_projection(pcdl[:, :4])
        ranger, _, _, _ = range_projection(pcdr[:, :4])
        rangec, _, _, _ = range_projection(pcdc[:, :4])
        data = DriftDataset.preproces(rangel, rangec, ranger, drift)
        return data

    def _create_data_list(self):
        """Create a datalist from self.data_dir"""
        data_list = []

        with open(self.filenames, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            # pcd, range image, count image, pickle
            for itr, row in enumerate(tqdm(reader, desc="DataLoader")):
                data_list.append(row)
        return data_list

    def preproces(rangel, rangec, ranger, drift, has_label=True, crop=True):
        """Preprocess data before training and testing"""

        #######
        if has_label:
            anchor = rangec
            if drift[0] < drift[-1]:
                positive, negative = rangel, ranger
            else:
                positive, negative = ranger, rangel
        else:
            anchor, positive, negative = rangec, rangel, ranger # Dosen't make difference
        ### Normalise and invert
        anchor = (1 - (anchor / 45.0)) * (anchor > 0)
        positive = (1 - (positive / 45.0)) * (positive > 0)
        negative = (1 - (negative / 45.0)) * (negative > 0)
        ######

        #######
        # Cropping
        # CROPS = [ 
        #     [(9, 14), (330, 562)],
        #     [(9, 13), (1255, 1445)],
        #     [(10, 14), (1515, 1725)],
        #     [(10, 14), (120, 330)],
            
        #     [(9, 12), (800, 930)],
        #     [(10, 13), (1100, 1233)],

        #     [(9, 11), (10, 110)],
        #     [(13, 15), (1650, 1755)],
        #     [(10, 12), (565, 665)],
        #     [(9, 11), (1100, 1200)],
        # ]
        # num_crops = np.random.choice([0, 2, 3, 4])
        # cropids = np.random.choice(a=len(CROPS), size=num_crops, replace=False)

        # for cropid in cropids: 
        #     crop = CROPS[cropid]
        #     anchor[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] = 0
        #     positive[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] = 0
        #     negative[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] = 0

        # -------------
        SMALL_CROPS = [
            # [(10, 12), (200, 300)],
            # [(13, 15), (250, 350)],
            # [(10, 12), (600, 800)],
            # [(13, 15), (650, 850)],
            # [(10, 12), (1400, 1600)],
            # [(13, 15), (1450, 1650)],

            [(10, 14), (100, 150)],
            [(11, 15), (150, 200)],
            [(10, 14), (350, 400)],
            [(11, 15), (250, 300)],
            [(10, 14), (800, 850)],
            [(10, 14), (600, 650)],
            [(10, 14), (500, 550)],
            [(11, 15), (850, 900)],
            [(11, 15), (1000, 1050)],
            [(10, 14), (1400, 1450)],
            [(10, 14), (1200, 1250)],
            [(10, 14), (1600, 1650)],
            [(11, 15), (1700, 1750)],

        ]

        # # num_crops = np.random.choice([4, 6, 8, 10, 12])
        # num_crops = len(SMALL_CROPS)
        # cropids = np.random.choice(a=len(SMALL_CROPS), size=num_crops, replace=False)

        # for cropid in cropids: 
        #     crop = SMALL_CROPS[cropid]
        #     anchor[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] = 0.7
        #     positive[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] = 0.7
        #     negative[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] = 0.7

        #######
        
        data = {
            'anchor': torch.tensor(anchor).unsqueeze(0).float(),
            'positive': torch.tensor(positive).unsqueeze(0).float(),
            'negative': torch.tensor(negative).unsqueeze(0).float(),
        }

        return data

    def get_split_sizes(self, train_percent, test_percent):
        """Handle size mismatches due to percentage multiplication"""

        train_size, test_size = [
            int(self.len * train_percent), int(self.len * test_percent)]
        if train_size + test_size > self.len:
            train_size -= train_size - (self.len - test_size)
        elif train_size + test_size < self.len:
            test_size += (self.len - train_size) - test_size
        return train_size, test_size


def test():
    # Hyperparams
    # batch_size = 32
    # # 40500
    # data_dir = 'data/diverse-data/'
    # LD = LidarDataset(data_dir)
    # dataloader = DataLoader(LD, batch_size=batch_size, shuffle=True)
    # # print(dataloader.dataset)
    # test_out = next(iter(dataloader))
    # print('test_out shape: ', test_out.shape)
    # print('To load next batch: test_out = next(iter(dataloader))')
    # breakpoint()

    batch_size = 32
    # 40500
    filenames = 'data/drift/training/filenames_comb.csv'
    DD = DriftDataset(filenames)
    dataloader = DataLoader(DD, batch_size=batch_size, shuffle=True)

    # # Snippet for data distribution
    # total, positive = 0, 0
    # for entry in dataloader:
    #     y = entry['y']
    #     positive += torch.sum(y).float()
    #     total += y.shape[0]
    # print('Total, Positive: ', total, positive)
    
    test_out = next(iter(dataloader))
    # print('test_out shape: ', test_out['ri'].shape)
    print('To load next batch: test_out = next(iter(dataloader))')
    breakpoint()


if __name__ == '__main__':
    test()
