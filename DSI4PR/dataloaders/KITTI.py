import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from tools.transform import get_filenames, get_transforms
from tools.range_image import loadCloudFromBinary, createRangeImage



def get_dataloader(filenames, mode, CFG):
    transforms = get_transforms(mode=mode, size=CFG.size)
    dataset = CILPDataset(
        transforms=transforms,
        CFG=CFG,
        filenames=filenames,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def get_poses(eval_sequence, CFG):
     # get all poses in training
        pose_file = CFG.data_path + '/' + eval_sequence + '/pose.txt'

        poses = pd.read_csv(pose_file, header=None,
                        delim_whitespace=True).to_numpy()

        translation_poses = poses[:, [3, 7, 11]]
        #return translation_poses

        indices= {}
        for i ,poses in enumerate(poses):
            indices[i] = poses[[3, 7, 11]]
        
    
        return translation_poses, indices
class CILPDataset(torch.utils.data.Dataset):

    # load from sequence number
    def __init__(self, transforms, CFG, filenames=[], sequences=[]):
        if(len(filenames) != 0):
            self.data_filenames = filenames
        else:
            self.data_filenames = get_filenames(sequences, CFG.data_path)
        self.transforms = transforms

        self.data_path = CFG.data_path
        self.CFG = CFG

    def __getitem__(self, idx):
        item = {}
        seq = self.data_filenames[idx].split('/')[0]
        instance = self.data_filenames[idx].split('/')[1]

        depth = cv2.imread(f"{self.data_path}/{seq}/depth/{instance}.png")

        lidar_points = loadCloudFromBinary(
            f"{self.data_path}/{seq}/velodyne/{instance}.bin")

        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        depth = self.transforms(image=depth)['image']
        item['depth_image'] = torch.tensor(depth).permute(2, 0, 1).float()

        lidar_image = createRangeImage(lidar_points, self.CFG.crop_distance, self.CFG.distance_threshold)


        lidar_image = self.transforms(image=lidar_image)['image']
        item['lidar_image'] = torch.tensor(
            lidar_image).permute(2, 0, 1).float()
        
      
        return item

    def __len__(self):
        return len(self.data_filenames)

    def flush(self):
        pass
