from torch.utils import data
from nn.dataset import LidarDataset
import numpy as np


def nearby_inifo(idx, data_list):
    print('\n\n\n##########')
    rel_diff = []
    for i in range(idx-20, idx+20):
        if data_list[idx].range_image_file == data_list[i].range_image_file:
            data = data_list[i]
            print(data)
            print( (data.tf - data.rf)/ data.rf)
            rel_diff.append((data.tf - data.rf)/ data.rf)

    rel_diff = np.array(rel_diff)
    print('########')
    print(rel_diff.min(), rel_diff.max())



def main():
    lidarDataset = LidarDataset('data/i2i-data/')
    data_list = lidarDataset.data_list
    while True:
        idx = int(input('Give Idx: '))
        nearby_inifo(idx, data_list)
    

if __name__=='__main__':
    main()