import numpy as np
import os
from osgeo import gdal
import torch

def fetchImages(directory):
    img = list()
    dist = list()
    for root, _, files in os.walk(os.path.abspath(directory)):
        for f in files:
            fl = f.lower()
            if fl.find('dist')!=-1:
                dist.append(os.path.join(root,f))
            else:
                img.append(os.path.join(root,f))
    return img,dist

def loadImage(imagepath):
    raster = gdal.Open(imagepath)
    if raster is None:
        return None
    band = raster.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float)
    raster = None
    return data

def loadDir(directory):
    ret = dict()
    for d in os.listdir(directory):
        path = os.path.join(directory,d)
        if os.path.isdir(path):
            ret[int(d)] = path
    return ret

class CrossViewDataset:
    def __init__(self, dirs_dict, useDist, train=True, transform=None):
        self.transform = transform
        self.train = train

        labels = list()
        data = list()
        for i,directory in dirs_dict.items():
            imgs,dist = fetchImages(directory)
            if useDist:
                imgs = dist
            for img in imgs:
                t = loadImage(img)
                if t is not None:
                    data.append(t)
                    labels.append(i)            
        
        self.targets = torch.as_tensor(labels)
        self.data = torch.as_tensor(data)
        
        if train:
            self.train_labels = torch.as_tensor(labels)
            self.train_data = torch.as_tensor(data)
        else:
            self.test_labels = torch.as_tensor(labels)
            self.test_data = torch.as_tensor(data)
    def __len__(self):
        return len(self.targets)

def loadCrossViewDataset(directory, useDist, train = True, transform = None):
    dirs_dict = loadDir(directory)
    return CrossViewDataset(dirs_dict,useDist,train,transform)