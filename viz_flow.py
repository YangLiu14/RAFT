import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.frame_utils import readFlow


DEVICE = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# def load_image(imfile):
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     return img[None].to(DEVICE)


def load_image(imfile):
    img = Image.open(imfile)

    size = 1024, 1024
    img.thumbnail(size, Image.ANTIALIAS)  # Downsize the image
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# def viz(img, flo):
#     img = img[0].permute(1, 2, 0).cpu().numpy()
#     # flo = flo[0].permute(1, 2, 0).cpu().numpy()
#
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([img, flo], axis=0)
#
#     # import matplotlib.pyplot as plt
#     # plt.imshow(img_flo / 255.0)
#     # plt.show()


def viz(img, flo, out_fpath):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    # flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()


def viz_opt_file(img, flo, out_fpath):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    # flo = flo[0].permute(1,2,0).cpu().numpy()

    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    # cv2.imwrite(out_fpath, img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite(out_fpath, img_flo[:, :, [2,1,0]])


def viz_flow_from_model(img, flo, out_fpath):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    # cv2.imwrite(out_fpath, img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite(out_fpath, img_flo[:, :, [2,1,0]])


def demo(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        flows = sorted(glob.glob(args.flow_dir + '*_up.flo'))

        idx = 0
        for imfile1, imfile2, flow_fn in zip(images[:-1], images[1:], flows):
            idx = idx + 1
            print(idx)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_up = readFlow(flow_fn)
            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            out_fname = flow_fn.split('/')[-1].replace('.flo', '.jpg')

            viz_opt_file(image1, flow_up, os.path.join(args.outdir, out_fname))
            # viz(image1, flow_up, os.path.join(args.outdir, out_fname))
            # viz_flow_from_model(image1, flow_up, os.path.join(args.outdir, out_fname))


if __name__ == '__main__':
    # BASE_DIR = "/mnt/raid/davech2y/liuyang/"
    BASE_DIR = "/storage/slurm/liuyang/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=BASE_DIR + "/model_weights/RAFT/raft-sintel.pth", help="restore checkpoint")
    # parser.add_argument('--model', default="models/raft-sintel.pth", help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--path', default=BASE_DIR + "/data/MOTS20/train/images/0002/", help="dataset for evaluation")
    parser.add_argument('--path', default=BASE_DIR + "/data/TAO/TAO_VAL/val/ArgoVerse/00c561b9-2057-358d-82c6-5b06d76cebcf/", help="dataset for evaluation")
    # parser.add_argument('--flow_dir', default=BASE_DIR + "/Optical_Flow/MOTS20_RAFT_sintel/images/0002/", help="directory storing the .flo files")
    parser.add_argument('--flow_dir', default=BASE_DIR + "/Optical_Flow/TaoVal_RAFT_sintel_downscaled/ArgoVerse/00c561b9-2057-358d-82c6-5b06d76cebcf/", help="directory storing the .flo files")
    parser.add_argument('--outdir', default=BASE_DIR + "/Optical_Flow/opt_viz_tmp/", help="visualization output dir")
    args = parser.parse_args()

    demo(args)
