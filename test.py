import torch
import src.net.HMSF as  net
import src.net.HMSF as HMSF
import os
import torchvision
import sys
import torch.nn as nn
import torch.nn.functional as F
import src.config.util as util
import src.data.data.util as data_util
import functools
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torchvision.transforms as t
from ptflops import get_model_complexity_info
import torch
from torch.autograd import Variable
import numpy as np
import  math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from src.net.EDSR import Net as edsr
datasets = ['Set5','Set14','B100','Urban100','manga109']
scale = str(3)
for dataset in datasets:
    print(dataset)
    test_dataset_folder = '/home/siat/文档/SR_Test_Datasets/' + scale +'x/'+dataset+'_down' +  scale + 'x'
    save1 = './result_' + scale +'x/' 
    save =   './result_' + scale +'x/' + dataset + '/'
    if not os.path.exists(save1 ):
        os.mkdir(save1)

    if not os.path.exists(save ):
        os.mkdir(save)


    modname = scale + 'x'

    model =  net.HMSF()
    model.load_state_dict(torch.load('./' + modname + '.pth'), strict=True)

    print(model)
    device = 'cuda:0'
    model.eval() 
    model=model.to(device)
    
    ext = '*.bmp'
    if dataset =='Set5' or dataset == 'Set14' or dataset == 'hot':
        ext='*.bmp'
    lists = glob.glob(os.path.join(test_dataset_folder, ext))
    lists.sort()

    sum = 0
    for image_name in lists:

        print(image_name)

        im_l_y = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    


        im_input = im_l_y/255.
        im_input = torch.Tensor(im_input).unsqueeze(0)
        im_input = im_input[:, :, :, [2, 1, 0]]

        im_input = torch.from_numpy(np.ascontiguousarray(np.transpose(im_input, (0, 3, 1, 2)))).float().to(device)
        




        with torch.no_grad():

            output = model(im_input)
 
            output = util.tensor2img(output)
        
        cv2.imwrite(save +image_name.split('/')[-1],output)

    # print("average FPS:", sum/len(lists))

print("*********************************************************************************")

macs, params = get_model_complexity_info(model, (3, 1280 // int(scale), 720 // int(scale)), as_strings=True,
                                        print_per_layer_stat=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))