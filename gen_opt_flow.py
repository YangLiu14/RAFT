import sys
sys.path.append('core')

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import glob
import numpy as np
import torch
import tqdm
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from utils.frame_utils import writeFlow


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!");
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def opt_flow_estimation(args):
    """
    args.path: path to the directory of the dataset that contains the images.
        - base_dir/
            - ArgoVerse/
                - video1/
                    - frame1.jpg
                    - frame2.jpg
                - video2/
            - BDD/
            - Charades/
            - LaSOT/
            - YFCC100M/
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        base_dir = args.path
        data_srcs = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(base_dir, '*')))]
        if args.datasrc:
            data_srcs = [args.datasrc]

        for data_src in data_srcs:
            print("Processing", data_src)
            videos = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(base_dir, data_src, '*')))]
            for idx, video in enumerate(tqdm.tqdm(videos)):
                fpath = os.path.join(base_dir, data_src, video)

                images = glob.glob(os.path.join(fpath, '*.png')) + \
                         glob.glob(os.path.join(fpath, '*.jpg'))

                images = sorted(images)
                for imfile1, imfile2 in zip(images[:-1], images[1:]):
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)

                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                    # Store the flow vector
                    flow_fname = imfile1.split("/")[-1].replace(".jpg", ".flo")
                    flow_up_fname = flow_fname.split(".")
                    flow_up_fname[0] = flow_up_fname[0] + "_up"
                    flow_up_fname = ".".join(flow_up_fname)
                    flow_low_fname = flow_fname.split(".")
                    flow_low_fname[0] = flow_low_fname[0] + "_low"
                    flow_low_fname = ".".join(flow_low_fname)

                    up_fname = os.path.join(args.outdir, data_src, video, flow_up_fname)
                    low_fname = os.path.join(args.outdir, data_src, video, flow_low_fname)
                    if not os.path.exists(os.path.join(args.outdir, data_src, video)):
                        os.makedirs(os.path.join(args.outdir, data_src, video))

                    flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
                    flow_low = flow_low[0].permute(1, 2, 0).cpu().numpy()

                    # writeFlowFile(up_fname, flow_up)
                    # writeFlowFile(low_fname, flow_low)
                    writeFlow(up_fname, flow_up)
                    writeFlow(low_fname, flow_low)
                    # viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--outdir', help="output directory for optical flow")
    parser.add_argument('--datasrc', help="which datasrc to process")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    opt_flow_estimation(args)

# Example command
"""
BASE_DAVE = /mnt/raid/davech2y/liuyang/
python gen_opt_flow.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth --path /mnt/raid/davech2y/liuyang/data/TAO/frames/val/ --outdir /mnt/raid/davech2y/liuyang/Optical_Flow/TaoVal_RAFT_sintel/

python gen_opt_flow.py --model /mnt/raid/davech2y/liuyang/model_weights/RAFT/raft-sintel.pth --path /mnt/raid/davech2y/liuyang/data/MOTS20/train/ --outdir /mnt/raid/davech2y/liuyang/Optical_Flow/MOTS20_RAFT_sintel/
"""