import os
import argparse
import cv2
import numpy as np
import torch
import timm

import test
from utils.data_utils import get_loader
from models_swin.ms_swin_transformer import *
from models_swin.swin_transformer import SwinTransformer
import models_swin.ms_backup as ms_b

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='../../datasets/cubbirds/test/',
                        help='Input image path')
    parser.add_argument('--image-class', type=int, default=1, help='choose image class in a subdir of image path')

    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')

    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    
    parser.add_argument('--stage', type=int, default=-1, help='choosing **cam stage')
    parser.add_argument('--block', type=int, default=-1, help='choosing **cam stage')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


class ReshapeT:
    def __init__(self, height=14, width=14):
        self.height = height
        self.width = width
    
    def reshape_transform(self, tensor):
        result = tensor.reshape(tensor.size(0), 
            self.height, self.width, tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
        
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM, 
         "scorecam": ScoreCAM, 
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

#     model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
#     model_ckpt = torch.load('output/sample_run_swin_t_no_checkpoint.bin')
    model_ckpt = torch.load('output/sample_run_swin_b_no_checkpoint.bin')
    
#     model=MSSwinTransformer(img_size=448, num_classes=200, num_feature_layers=1, detail_features=True)
    model=SwinTransformer(img_size=448, num_classes=200, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
#     base
#     model=ms_b.MSSwinTransformer(img_size=448, num_classes=200, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
    
    model.load_state_dict(model_ckpt['model'], strict=False)
    
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layer = model.layers[args.stage].blocks[args.block].norm1
    height = int(112/(2**((args.stage+4)%4)))
    width = int(112/(2**((args.stage+4)%4)))
    reshape_t = ReshapeT(height=height, width=width)

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model, 
                               target_layer=target_layer,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_t.reshape_transform)
    
    confusion = []
    imgclass = os.listdir(args.image_path)
    imgclass.sort()
    for i in range(len(imgclass)):
        imgclass_path=f'./cam/{imgclass[i]}'
        if not os.path.exists(imgclass_path):
            os.makedirs(imgclass_path)
        imglist = os.listdir(os.path.join(args.image_path, imgclass[i]))
        imglist.sort()
        print("lenlist={}".format(len(imglist)))
        
        for j in range(len(imglist)):
            imgname = imglist[j]
            rgb_img = cv2.imread(os.path.join(args.image_path, imgclass[i], imgname), 1)
            if rgb_img is None:
                print(os.path.join(args.image_path,imgname))
            else:
                print(imgname)
                rgb_img = rgb_img[:, :, ::-1]
                rgb_img = cv2.resize(rgb_img, (448, 448))
                rgb_img = np.float32(rgb_img) / 255
                input_tensor = preprocess_image(rgb_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                if args.use_cuda:
                    input_tensor=input_tensor.cuda()

                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                target_category = None

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32

                y=model(input_tensor)
                confusion.append(y.detach().cpu().numpy())
                print(torch.argmax(y))
        #         print(y.shape)
                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    eigen_smooth=args.eigen_smooth,
                                    aug_smooth=args.aug_smooth)



                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                cv2.imwrite(os.path.join(imgclass_path, f'{i:03}_{args.stage}_{args.block}_' + imgname[:-4] + f'_{args.method}_cam.jpg'), cam_image)
                
    confusion = np.array(confusion)
    np.save('./confusion_base.npy', confusion)