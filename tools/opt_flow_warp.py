import os, sys
sys.path.append("..")
import glob
import argparse
import tqdm
import numpy as np
import cv2
import json
import shutil
from pycocotools.mask import encode, iou, area, decode, toBbox, merge
from PIL import Image
from time import time
from flow_utils import visulize_flow_file
from visualize import visualize_proposals, save_with_pascal_colormap
from core.utils.frame_utils import readFlow

# BASE_DIR = "/mnt/raid/davech2y/"
BASE_DIR = "/storage/slurm/"

def viz_mask(masks, image_size, out_fn):
    png = np.zeros(image_size, dtype=int)
    for idx, mask in enumerate(masks):
        png[mask.astype("bool")] = idx + 2
        save_with_pascal_colormap(out_fn, png)


# ============================================
# Optical-flow related helper functions
# ============================================
def warp_flow(img, flow, binarize=True):
    """
    Use the given optical-flow vector to warp the input image/mask in frame t-1,
    to estimate its shape in frame t.
    :param img: (H, W, C) numpy array, if C=1, then it's omissible. The image/mask in previous frame.
    :param flow: (H, W, 2) numpy array. The optical-flow vector.
    :param binarize:
    :return: (H, W, C) numpy array. The warped image/mask.
    """
    h, w = flow.shape[:2]
    # img preprocessing: downscaled the mask
    img = Image.fromarray(img)
    size = max(h, w), max(h, w)
    img.thumbnail(size, Image.ANTIALIAS)  # Downsize the image
    img = (np.array(img)).astype(np.uint8)

    # # TEST
    # unique, count = np.unique(img, return_counts=True)
    # cv2.imwrite("schrinked_mask.png", img * 255)
    # # END of TEST

    # img = (img / 255).astype(np.uint8)
    # END of img preprocessing

    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    if binarize:
        res = np.equal(res, 1).astype(np.uint8)
    return res


def warp_proposals_per_frame(frame_fn: str, flow_fn: str, json_out_dir, file_type, visualize=False):
    """
    Extract masks from the given frame (json), wrap them as "forward_segmentation"
    and store them in the new json file together with the original segmentation.

    :param frame_fn: json file path of the current frame.
    :param flow_fn: .flo file path of the optical-flow vector.
    The vector is computed between this frame and the next frame.
    :param visualize: if True, store the `orginal mask` and the `warped mask` as png during the process.
    """

    if file_type == ".json":
        with open(frame_fn, 'r') as f:
            proposals = json.load(f)  # list of dict
    elif file_type == ".npz":
        npz_file = np.load(frame_fn, allow_pickle=True)
        proposals = npz_file['arr_0'].tolist()
    else:
        raise Exception("Unrecognized file type.")

    flow = readFlow(flow_fn)

    all_masks = []
    all_warps = []
    for idx, prop in enumerate(proposals):
        # numpy array of shape (H, W)
        mask = decode(prop['instance_mask'])
        warped_mask = warp_flow(mask, flow)
        all_masks.append(mask)
        all_warps.append(warped_mask)
        # store warped mask
        warp_enc = encode(np.array(warped_mask[:, :, np.newaxis], order='F'))[0]
        warp_enc['counts'] = warp_enc['counts'].decode(encoding="utf-8")
        proposals[idx]['forward_segmentation'] = warp_enc

    folder_name = '/'.join(frame_fn.split('/')[-3:-1])
    fn = frame_fn.split('/')[-1].replace('.json', '')

    if not os.path.exists(json_out_dir + "/" + folder_name):
        os.makedirs(json_out_dir + "/" + folder_name)

    with open(os.path.join(json_out_dir, folder_name, fn + '.json'), 'w') as fout:
        json.dump(proposals, fout)

    if visualize:
        # Determine the store path for mask visulization
        mask_store_dir = os.path.join(BASE_DIR, "liuyang/Optical_Flow/mask_visualization/tao_val", folder_name) + '/'

        if not os.path.exists(mask_store_dir):
            os.makedirs(mask_store_dir)
        if proposals != "":
            image_size = proposals[0]['instance_mask']['size']
            viz_mask(all_masks, image_size, mask_store_dir + fn+'_mask'+'.png')
            viz_mask(all_warps, image_size, mask_store_dir + fn+'_warp'+'.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ROOT_DIR = BASE_DIR + "/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/"
    parser.add_argument('--prop_dir', type=str,
                        # default=ROOT_DIR + "boxNMS/_objectness_tmp/",
                        default=BASE_DIR + "/liuyang/data/MOTS20/MOTS20_provided_detections/MOTSChallenge/",
                        help='Location where the segmentations are')
    parser.add_argument('--opt_flow_dir', type=str,
                        # default=BASE_DIR + "/liuyang/Optical_Flow/RAFT_sintel_tmp/",
                        default=BASE_DIR + "/liuyang/Optical_Flow/MOTS20_RAFT_sintel/",
                        help='Location where the optical flow vectors are stored')
    parser.add_argument('--out_dir', type=str,
                        default=ROOT_DIR + "/optical_flow_output/",
                        # default="/nfs/volume-411-3/liuyang/bdd_100k/VAL/2_optical_flow/jsons/",
                        # default="/nfs/volume-411-3/liuyang/bdd_100k/TEST/2_optical_flow/jsons/",
                        help='Output directory')
    parser.add_argument('--datasrc', nargs='+', type=str, help='Process only specific data source')
    parser.add_argument('--file_type', default=".json", type=str, help='.json or .npz')
    parser.add_argument('--visualize', action='store_true', help='whether to visualize the warped masks')

    args = parser.parse_args()

    all_proposals_dir = args.prop_dir

    opt_flow_dir = args.opt_flow_dir
    out = args.out_dir
    print(">>>> proposals    from:", all_proposals_dir)
    print(">>>> optical flow from:", opt_flow_dir)
    print(">>>> output         to:", out)

    data_srcs = ["ArgoVerse", "BDD", "Charades", "LaSOT", "YFCC100M"]
    if args.datasrc:
        data_srcs = args.datasrc
    for datasrc in data_srcs:
        # Folder names of all the video sequence
        print("Processing", datasrc)
        video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(all_proposals_dir, datasrc, '*')))]

        for idx, video in enumerate(video_names):
            print("Processing", (datasrc + '/' + video))
            # list of json file names in the current video sequence
            video_path = os.path.join(all_proposals_dir, datasrc, video) + '/'
            flow_path = os.path.join(opt_flow_dir, datasrc, video) + '/'

            frames = sorted(glob.glob(video_path + '*'))
            flows = sorted(glob.glob(flow_path + '*_up.flo'))
            assert len(frames) - 1 == len(flows), "Inconsistent file amount between proposals and optical-flow vectors"

            # root_dir = '/'.join(video_path.split('/')[:-2])
            folder_name = '/'.join(video_path.split('/')[-3:-1])

            out_dir = out + "/" + folder_name + "/"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            t = time()
            for frame_fn, flow_fn in tqdm.tqdm(zip(frames, flows), total=len(flows)):
                warp_proposals_per_frame(frame_fn, flow_fn, out, file_type=args.file_type, visualize=args.visualize)

            if args.file_type == ".json":
                # add the last proposal directly to output file:
                fn = frames[-1].split("/")[-1]
                shutil.copyfile(frames[-1], out_dir + fn)
            elif args.file_type == ".npz":
                fn = frames[-1].split("/")[-1]
                proposals = np.load(frames[-1], allow_pickle=True)['arr_0'].tolist()
                with open(os.path.join(out, folder_name, fn + '.json'), 'w') as fout:
                    json.dump(proposals, fout)

            print(str(idx), 'video', video, 'finished in', time() - t, 'seconds.', len(flows), 'flows at',
                  (time() - t) / (len(flows) - 1), 'per image.')

