#!/usr/bin/env python3

import glob
import numpy as np
import tqdm
import os
import shutil
from PIL import Image
import cv2
from imageio import imread

pascal_colormap = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0,
    0.7529, 0.2510, 0,
    0.2510, 0.7529, 0,
    0.7529, 0.7529, 0,
    0.2510, 0.2510, 0.5020,
    0.7529, 0.2510, 0.5020,
    0.2510, 0.7529, 0.5020,
    0.7529, 0.7529, 0.5020,
    0, 0, 0.2510,
    0.5020, 0, 0.2510,
    0, 0.5020, 0.2510,
    0.5020, 0.5020, 0.2510,
    0, 0, 0.7529,
    0.5020, 0, 0.7529,
    0, 0.5020, 0.7529,
    0.5020, 0.5020, 0.7529,
    0.2510, 0, 0.2510,
    0.7529, 0, 0.2510,
    0.2510, 0.5020, 0.2510,
    0.7529, 0.5020, 0.2510,
    0.2510, 0, 0.7529,
    0.7529, 0, 0.7529,
    0.2510, 0.5020, 0.7529,
    0.7529, 0.5020, 0.7529,
    0, 0.2510, 0.2510,
    0.5020, 0.2510, 0.2510,
    0, 0.7529, 0.2510,
    0.5020, 0.7529, 0.2510,
    0, 0.2510, 0.7529,
    0.5020, 0.2510, 0.7529,
    0, 0.7529, 0.7529,
    0.5020, 0.7529, 0.7529,
    0.2510, 0.2510, 0.2510,
    0.7529, 0.2510, 0.2510,
    0.2510, 0.7529, 0.2510,
    0.7529, 0.7529, 0.2510,
    0.2510, 0.2510, 0.7529,
    0.7529, 0.2510, 0.7529,
    0.2510, 0.7529, 0.7529,
    0.7529, 0.7529, 0.7529,
    0.1255, 0, 0,
    0.6275, 0, 0,
    0.1255, 0.5020, 0,
    0.6275, 0.5020, 0,
    0.1255, 0, 0.5020,
    0.6275, 0, 0.5020,
    0.1255, 0.5020, 0.5020,
    0.6275, 0.5020, 0.5020,
    0.3765, 0, 0,
    0.8784, 0, 0,
    0.3765, 0.5020, 0,
    0.8784, 0.5020, 0,
    0.3765, 0, 0.5020,
    0.8784, 0, 0.5020,
    0.3765, 0.5020, 0.5020,
    0.8784, 0.5020, 0.5020,
    0.1255, 0.2510, 0,
    0.6275, 0.2510, 0,
    0.1255, 0.7529, 0,
    0.6275, 0.7529, 0,
    0.1255, 0.2510, 0.5020,
    0.6275, 0.2510, 0.5020,
    0.1255, 0.7529, 0.5020,
    0.6275, 0.7529, 0.5020,
    0.3765, 0.2510, 0,
    0.8784, 0.2510, 0,
    0.3765, 0.7529, 0,
    0.8784, 0.7529, 0,
    0.3765, 0.2510, 0.5020,
    0.8784, 0.2510, 0.5020,
    0.3765, 0.7529, 0.5020,
    0.8784, 0.7529, 0.5020,
    0.1255, 0, 0.2510,
    0.6275, 0, 0.2510,
    0.1255, 0.5020, 0.2510,
    0.6275, 0.5020, 0.2510,
    0.1255, 0, 0.7529,
    0.6275, 0, 0.7529,
    0.1255, 0.5020, 0.7529,
    0.6275, 0.5020, 0.7529,
    0.3765, 0, 0.2510,
    0.8784, 0, 0.2510,
    0.3765, 0.5020, 0.2510,
    0.8784, 0.5020, 0.2510,
    0.3765, 0, 0.7529,
    0.8784, 0, 0.7529,
    0.3765, 0.5020, 0.7529,
    0.8784, 0.5020, 0.7529,
    0.1255, 0.2510, 0.2510,
    0.6275, 0.2510, 0.2510,
    0.1255, 0.7529, 0.2510,
    0.6275, 0.7529, 0.2510,
    0.1255, 0.2510, 0.7529,
    0.6275, 0.2510, 0.7529,
    0.1255, 0.7529, 0.7529,
    0.6275, 0.7529, 0.7529,
    0.3765, 0.2510, 0.2510,
    0.8784, 0.2510, 0.2510,
    0.3765, 0.7529, 0.2510,
    0.8784, 0.7529, 0.2510,
    0.3765, 0.2510, 0.7529,
    0.8784, 0.2510, 0.7529,
    0.3765, 0.7529, 0.7529,
    0.8784, 0.7529, 0.7529,
    0, 0.1255, 0,
    0.5020, 0.1255, 0,
    0, 0.6275, 0,
    0.5020, 0.6275, 0,
    0, 0.1255, 0.5020,
    0.5020, 0.1255, 0.5020,
    0, 0.6275, 0.5020,
    0.5020, 0.6275, 0.5020,
    0.2510, 0.1255, 0,
    0.7529, 0.1255, 0,
    0.2510, 0.6275, 0,
    0.7529, 0.6275, 0,
    0.2510, 0.1255, 0.5020,
    0.7529, 0.1255, 0.5020,
    0.2510, 0.6275, 0.5020,
    0.7529, 0.6275, 0.5020,
    0, 0.3765, 0,
    0.5020, 0.3765, 0,
    0, 0.8784, 0,
    0.5020, 0.8784, 0,
    0, 0.3765, 0.5020,
    0.5020, 0.3765, 0.5020,
    0, 0.8784, 0.5020,
    0.5020, 0.8784, 0.5020,
    0.2510, 0.3765, 0,
    0.7529, 0.3765, 0,
    0.2510, 0.8784, 0,
    0.7529, 0.8784, 0,
    0.2510, 0.3765, 0.5020,
    0.7529, 0.3765, 0.5020,
    0.2510, 0.8784, 0.5020,
    0.7529, 0.8784, 0.5020,
    0, 0.1255, 0.2510,
    0.5020, 0.1255, 0.2510,
    0, 0.6275, 0.2510,
    0.5020, 0.6275, 0.2510,
    0, 0.1255, 0.7529,
    0.5020, 0.1255, 0.7529,
    0, 0.6275, 0.7529,
    0.5020, 0.6275, 0.7529,
    0.2510, 0.1255, 0.2510,
    0.7529, 0.1255, 0.2510,
    0.2510, 0.6275, 0.2510,
    0.7529, 0.6275, 0.2510,
    0.2510, 0.1255, 0.7529,
    0.7529, 0.1255, 0.7529,
    0.2510, 0.6275, 0.7529,
    0.7529, 0.6275, 0.7529,
    0, 0.3765, 0.2510,
    0.5020, 0.3765, 0.2510,
    0, 0.8784, 0.2510,
    0.5020, 0.8784, 0.2510,
    0, 0.3765, 0.7529,
    0.5020, 0.3765, 0.7529,
    0, 0.8784, 0.7529,
    0.5020, 0.8784, 0.7529,
    0.2510, 0.3765, 0.2510,
    0.7529, 0.3765, 0.2510,
    0.2510, 0.8784, 0.2510,
    0.7529, 0.8784, 0.2510,
    0.2510, 0.3765, 0.7529,
    0.7529, 0.3765, 0.7529,
    0.2510, 0.8784, 0.7529,
    0.7529, 0.8784, 0.7529,
    0.1255, 0.1255, 0,
    0.6275, 0.1255, 0,
    0.1255, 0.6275, 0,
    0.6275, 0.6275, 0,
    0.1255, 0.1255, 0.5020,
    0.6275, 0.1255, 0.5020,
    0.1255, 0.6275, 0.5020,
    0.6275, 0.6275, 0.5020,
    0.3765, 0.1255, 0,
    0.8784, 0.1255, 0,
    0.3765, 0.6275, 0,
    0.8784, 0.6275, 0,
    0.3765, 0.1255, 0.5020,
    0.8784, 0.1255, 0.5020,
    0.3765, 0.6275, 0.5020,
    0.8784, 0.6275, 0.5020,
    0.1255, 0.3765, 0,
    0.6275, 0.3765, 0,
    0.1255, 0.8784, 0,
    0.6275, 0.8784, 0,
    0.1255, 0.3765, 0.5020,
    0.6275, 0.3765, 0.5020,
    0.1255, 0.8784, 0.5020,
    0.6275, 0.8784, 0.5020,
    0.3765, 0.3765, 0,
    0.8784, 0.3765, 0,
    0.3765, 0.8784, 0,
    0.8784, 0.8784, 0,
    0.3765, 0.3765, 0.5020,
    0.8784, 0.3765, 0.5020,
    0.3765, 0.8784, 0.5020,
    0.8784, 0.8784, 0.5020,
    0.1255, 0.1255, 0.2510,
    0.6275, 0.1255, 0.2510,
    0.1255, 0.6275, 0.2510,
    0.6275, 0.6275, 0.2510,
    0.1255, 0.1255, 0.7529,
    0.6275, 0.1255, 0.7529,
    0.1255, 0.6275, 0.7529,
    0.6275, 0.6275, 0.7529,
    0.3765, 0.1255, 0.2510,
    0.8784, 0.1255, 0.2510,
    0.3765, 0.6275, 0.2510,
    0.8784, 0.6275, 0.2510,
    0.3765, 0.1255, 0.7529,
    0.8784, 0.1255, 0.7529,
    0.3765, 0.6275, 0.7529,
    0.8784, 0.6275, 0.7529,
    0.1255, 0.3765, 0.2510,
    0.6275, 0.3765, 0.2510,
    0.1255, 0.8784, 0.2510,
    0.6275, 0.8784, 0.2510,
    0.1255, 0.3765, 0.7529,
    0.6275, 0.3765, 0.7529,
    0.1255, 0.8784, 0.7529,
    0.6275, 0.8784, 0.7529,
    0.3765, 0.3765, 0.2510,
    0.8784, 0.3765, 0.2510,
    0.3765, 0.8784, 0.2510,
    0.8784, 0.8784, 0.2510,
    0.3765, 0.3765, 0.7529,
    0.8784, 0.3765, 0.7529,
    0.3765, 0.8784, 0.7529,
    0.8784, 0.8784, 0.7529]

detectron_colormap = [
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    0.167, 0.000, 0.000,
    0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.286, 0.286, 0.286,
    0.429, 0.429, 0.429,
    0.571, 0.571, 0.571,
    0.714, 0.714, 0.714,
    0.857, 0.857, 0.857,
    1.000, 1.000, 1.000
]


def draw_mask(im, mask, alpha=0.5, color=None):
    colmap = (np.array(pascal_colormap) * 255).round().astype("uint8").reshape(256, 3)

    if color is None:
        color = detectron_colormap[np.random.choice(len(detectron_colormap))][::-1]
    else:
        while color >= 255:
            color = color - 254
        color = colmap[color]

    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im


def save_jpg(masks, t, image_dir, viz_dir, mask_ids, name=None):
    if name is not None:
        viz_dir = viz_dir % name

    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    img = imread(image_dir)
    img = img[:, :, :3]
    for i, (idx, mask) in enumerate(zip(mask_ids, masks)):
        img = draw_mask(img, mask, color=idx)

    cv2.imwrite(viz_dir + '/' + str(t + 1).zfill(5) + '.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_with_pascal_colormap(img_dir, img):
    colmap = (np.array(pascal_colormap) * 255).round().astype("uint8")
    palimage = Image.new('P', (16, 16))
    palimage.putpalette(colmap)
    im = Image.fromarray(np.squeeze(img.astype("uint8")))
    im2 = im.quantize(palette=palimage)
    im2.save(img_dir)


def visualize_tracklets(tracklets, all_props, image_size, output_directory, name=None):
    if name is not None:
        output_directory = output_directory % name

    if os.path.exists(output_directory):  # os.path.exists(output_directory % name):
        shutil.rmtree(output_directory)

    png = np.zeros(image_size, dtype=int)

    if len(tracklets) > 0:
        for t, props in enumerate(all_props):
            if len(props) > 0:
                props_to_use = tracklets[:, t]
                props_to_use_ind = np.where(tracklets[:, t] != -1)[0].tolist()
                for j, i in enumerate(props_to_use_ind):
                    png[props["mask"][props_to_use[i]].astype("bool")] = 2
                    tracklet_directory = output_directory + 'tracklet_' + str(i) + '/'
                    if not os.path.exists(tracklet_directory):
                        os.makedirs(tracklet_directory)
                    save_with_pascal_colormap(tracklet_directory + str(t + 1).zfill(5) + '.png', png)
                    png = np.zeros(image_size)


def visualize_proposals(proposals, image_size, output_directory, name=None):
    png = np.zeros(image_size, dtype=int)

    if name is not None:
        output_directory = output_directory % name

    for t, props in enumerate(proposals):
        directory = output_directory + 'time_' + str(t) + '/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        if len(props['seg']) > 0:
            for i in range(len(props['mask'])):
                png[props["mask"][i].astype("bool")] = 2
                save_with_pascal_colormap(directory + str(i).zfill(5) + '.png', png)
                png = np.zeros_like(props["mask"][0])
        else:
            save_with_pascal_colormap(directory + str(i).zfill(5) + '.png', png)


def convert_frames_to_video(pathIn, pathOut, origin_pathIn=None, fps=10):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if os.path.isfile(os.path.join(pathIn, f)) and f != '._.DS_Store']

    # for sorting the file names properly
    # files.sort(key=lambda x: int(x[5:-4]))
    files.sort()

    if origin_pathIn:
        origin_files = [f for f in os.listdir(origin_pathIn) if os.path.isfile(os.path.join(origin_pathIn, f)) and f != '._.DS_Store']
        origin_files.sort()

    for i in tqdm.tqdm(range(len(files))):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # print(filename)

        # combine original video frame with optical-flow visualization
        if origin_pathIn:
            orig_img = cv2.imread(origin_pathIn + "/" + origin_files[i])
            combined_img = np.vstack((orig_img, img))
            size = (width, 2*height)
        else:
            combined_img = img

        # inserting the frames into an image array
        # frame_array.append(img)
        frame_array.append(combined_img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__ == "__main__":
    print("Combine frames to video")
    # video_names = ["0001", "0006", "0007", "0012"]
    # video_names = ["0002", "0005", "0009", "0011"]
    fps_list = [30, 14, 30, 30]

    video_dir = "/nfs/cold_project/liuyang/mots1/tmp_lyuwei5/"
    video_names = [fn.split('/')[-1] for fn in sorted(glob.glob(os.path.join(video_dir, '*')))]
    # fps_list = [5] * len(video_names)

    count = 1
    for video_name, fps in zip(video_names, fps_list):
        print("PROCESS VIDEO {}: {}".format(count, video_name))
        pathIn = video_dir + video_name + "/"
        pathOut = video_dir + '/' + video_name + ".mp4"
        # origin_pathIn = "/nfs/project/lyuwei/data/KITTI_MOTS/data_tracking_image_2/training/image_02/" + video_name + "/"
        origin_pathIn = None
        convert_frames_to_video(pathIn, pathOut, fps=5)
        count += 1

    # image_dir = "/nfs/cold_project/liuyang/mots2020/unovost_output/training_ReID_premvos"