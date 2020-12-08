#!/usr/bin/env python3
"""process_gt.py: experimental: read gt mask into json"""
__author__ = "Yang Liu"
__email__ = "lander14@outlook.com"

import PIL.Image as Image
import numpy as np
import pycocotools.mask as rletools
import glob
import os
import time
import json
import tqdm
from pycocotools.mask import decode
from typing import Dict


class SegmentedObject:
  def __init__(self, mask, bbox, class_id, confidence):
    self.mask = mask
    self.bbox = bbox
    self.class_id = class_id
    self.conf = confidence


def load_txt(path):
    objects_per_frame = {}
    # track_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            # if frame not in track_ids_per_frame:
            #     track_ids_per_frame[frame] = set()
            # if int(fields[1]) in track_ids_per_frame[frame]:
            #     assert False, "Multiple objects with track id " + fields[1] + " in frame " + fields[0]
            # else:
            #     track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[6])
            confidence = float(fields[5])
            if not (class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {'size': [int(fields[7]), int(fields[8])], 'counts': fields[9].encode(encoding='UTF-8')}
            bbox = [float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4])]
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            # elif rletools.area(rletools.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0.0:
            #     assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge([combined_mask_per_frame[frame], mask], intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                bbox,
                class_id,
                confidence
            ))

    return objects_per_frame


def compute_bbox_from_mask(mask_rle: str):
    mask = decode(mask_rle)
    # Compute bbox
    pos = np.where(mask == 1)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    box = [float(xmin), float(ymin), float(xmax), float(ymax)]
    return box


def store_prop_in_json(object_per_frame: Dict, out_dir: str):
    for t, frame in enumerate(tqdm.tqdm(object_per_frame.keys())):
        out_json = []
        for prop in object_per_frame[frame]:
            if prop.conf < 0.1:
                continue
            # Construct one proposal
            proposal = dict()
            assert prop.class_id == 2
            proposal['category_id'] = 1  # to be consistent with Huangzhipeng's implementation
            proposal['bbox'] = prop.bbox
            proposal['score'] = prop.conf
            seg = dict()
            seg['size'] = prop.mask['size']
            seg['counts'] = prop.mask['counts'].decode(encoding="utf-8")
            proposal['instance_mask'] = seg
            # forward segmentation:
            # 当前帧的mask用光流投影转换到下一帧的mask
            fwd_seg = dict()
            fwd_seg['size'] = prop.mask['size']
            fwd_seg['counts'] = None
            proposal['forward_segmentation'] = fwd_seg
            proposal['backward_segmentation'] = None
            # ReID vector to be computed
            proposal['ReID'] = None

            # Append the proposal to output
            out_json.append(proposal)

        out_path = out_dir + '/' + str(t+1).zfill(6) + '.json'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_path, 'w') as fout:
            json.dump(out_json, fout)


if __name__ == "__main__":
    BASE_DIR = "/mnt/raid/davech2y/liuyang/"

    txt_dir = os.path.join(BASE_DIR, "data/MOTS20/MOTS20_provided_detections/MOTSChallenge/trainval/")
    out_basedir = os.path.join(BASE_DIR, "data/MOTS20/MOTS20_provided_detections/MOTSChallenge/")
    seq_names = glob.glob(txt_dir + "/0002*")
    seq_names.sort()
    for seq_name in seq_names:
        txt_path = seq_name
        object_per_frame = load_txt(txt_path)  # it's a dict

        dir_name = seq_name.split('/')[-1][:-4]
        out_dir = out_basedir + dir_name

        t = time.time()
        print("Start processing {}".format(dir_name))
        store_prop_in_json(object_per_frame, out_dir)
        print('Sequence', dir_name, 'finished in', time.time() - t, 'seconds.')

    print("Process complete")
