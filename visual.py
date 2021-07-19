import os
import argparse
import cv2
import numpy as np
import torch
import timm
import matplotlib.pyplot as plt

import test
from utils.data_utils import get_loader
from models_swin.ms_swin_transformer import *


# for CAM
from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image




# visualize feature map + channel correlation
def featuremap(f, num_batch=0, edge=10, start_ch=None):
    B, L, C = f.shape
    # L2 normalization for feature map
    n = f**2
    f1 = f/np.tile(np.sqrt(n.sum(axis=1)), (L,1,1)).transpose(1,0,2)
    # corr
    corr = np.einsum("jk, jm->km",f1[num_batch],f1[num_batch])
    H = int(np.sqrt(L))
    W = int(np.sqrt(L))
    f_ = f.reshape(B, H, W, C)
    
    f_ = f_[num_batch,:,:,:].reshape(H,W,C)
    fout = []
    if start_ch is None:
        ch_select = np.random.randint(0,C,size=(edge**2))
        for i in range(edge):
            fout.append(np.hstack([f_[:,:,j] for j in ch_select[edge*i:edge*(i+1)]]))
        fout = np.vstack(fout)    
    else:
        for i in range(edge):
            fout.append(np.hstack([f_[:,:,start_ch+j] for j in range(edge*i,edge*(i+1))]))
        fout = np.vstack(fout)
    return corr, fout

def v_featuremap(num_classes=200, layers=[2,2,6,2]):
    # setup
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    # args
    args = test.parse_option()
    # test.set_seed(args)

    # data
    _, test_loader = get_loader(args)
    # train_list = list(enumerate(train_loader))
    test_list = list(enumerate(test_loader))

    # model
    model_ckpt = torch.load('output/sample_run_swin_t_0.5_checkpoint.bin')
    model=MSSwinTransformer(img_size=448, num_classes=200, detail_features=True, num_feature_layers=len(layers))
    model.load_state_dict(model_ckpt['model'])
    model.eval()
    # x=torch.randn((16,3,448,448))
    x = test_list[0][1][0]
    f=model.forward_features(x)
    # f.shape

    layers = layers
    lf = model.layer_features
    for i in range(len(layers)):
        for j in range(layers[i]):
            stage = i
            block = j
            edge=6
            corr,f = featuremap(lf[stage][block], num_batch=0, edge=edge, start_ch=0)

            plt.figure()
            plt.suptitle('stage={0},block={1}'.format(stage, block), fontsize=14)
            plt.subplot(1, 2, 1)
            plt.imshow(corr)
            plt.title("channel correlation")
            plt.subplot(1, 2, 2)
            plt.imshow(f)
            plt.title("feature map")
            # plt.colorbar()
            plt.savefig('./visual/stage={0}_block={1}_edge{2}.png'.format(stage,block,edge),dpi=300)
            plt.show()
            
            
#####################################################################################################################
# CAM
def get_cam_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')

    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), 
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":
    v_featuremap()