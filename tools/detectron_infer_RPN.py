# coding: utf-8
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import json
from io import BytesIO as Bytes2Data
import numpy as np
import sys

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from core.rpn_generator import im_proposals
from core.rpn_generator import model_builder
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import utils.net as nu
import pycocotools.mask as mask_util

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class args: pass

args.cfg = '/data/tools/detectron/config/rpn_R-50-FPN_1x.yaml' # url: https://github.com/facebookresearch/Detectron/blob/master/configs/12_2017_baselines/rpn_R-50-FPN_1x.yaml
args.weights = '/data/tools/detectron/pretrain_model/model_final.pkl' # url: https://dl.fbaipublicfiles.com/detectron/35998814/12_2017_baselines/rpn_R-50-FPN_1x.yaml.08_06_03.Axg0r179/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl

import subprocess

def call(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = popen.communicate()
    return str(out)

merge_cfg_from_file(args.cfg)
cfg.TEST.WEIGHTS = args.weights
cfg.NUM_GPUS = 1
assert_and_infer_cfg()
model = infer_engine.initialize_model_from_cfg()
dummy_coco_dataset = dummy_datasets.get_coco_dataset()


def list_jpgs(DIR, ext='jpg'):
    return call('find %s -type f -name \"*.%s\"' % (DIR, ext)).split('\n')[:-1]

def infer(img_path):
    im = cv2.imread(img_path)
    with c2_utils.NamedCudaScope(0):
        boxes, scores = im_proposals(model, im)
    return im, boxes, scores
    
def vis(im, cls_boxes, cls_segms, cls_keyps, IMG_FILE, OUT_DIR, THRESH):
    out_name = os.path.join(OUT_DIR, '%s.jpg' % os.path.basename(IMG_FILE)[:-4])
    print('Processing %s -> %s' % (IMG_FILE, out_name))
    vis_utils.vis_one_image(
        im[:, :, ::-1],  # BGR -> RGB for visualization
        IMG_FILE[:-4],
        OUT_DIR,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_coco_dataset,
        box_alpha=0.8,
        show_class=False,
        thresh=THRESH,
        kp_thresh=2,
        ext='jpg'
    )

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class STHDataset():
    def __init__(self, source_path, txt):
        self.data = [VideoRecord(x.strip().split(' ')) for x in open(txt)]
        self.source_path = source_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        start = 1
        end = self.data[index].num_frames + 1
        video_path = os.path.join(self.source_path, self.data[index].path)
        img_lst = []
        for i in range(start, end):
            img_lst.append(os.path.join(video_path, '%05d.jpg'%i))
        return img_lst

def process_data(source_path, target_path, version, txt_file):
    data = STHDataset(source_path, 'data/something-something-{}/something-something-{}.txt'.format(version, txt_file))
    boxes_dict = {}
    for i in range(len(data)):
        video_boxes = []
        video_img_lst = data[i]
        for img_path in video_img_lst:
            im, boxes, scores = infer(img_path)
            if boxes is None:
                assert False, 'Empty boxes.'
            else:
                boxes[:,0::2] = boxes[:,0::2] / im.shape[1]
                boxes[:,1::2] = boxes[:,1::2] / im.shape[0]
                video_boxes.append(boxes.tolist())
        boxes_dict[data.data[i].path] = video_boxes
        if (i + 1) % 1000 == 0:
            print(i + 1)
    with open('{}/something-something-{}/region_proposal/something-something-{}.json'.format(target_path, version, txt_file),"w") as f:
        json.dump(boxes_dict, f)

def task_demo(DEMO_DIR, OUT_DIR, THRESH=0.8):
    files = list_jpgs(DEMO_DIR)
    for fname in files:
        im, boxes, scores = infer(fname)
        boxes = np.concatenate((boxes, np.expand_dims(scores, -1)), axis=-1)
        vis(im, boxes, None, None, fname, OUT_DIR, THRESH)

if __name__ == '__main__':
    # source_path = '/data/share/20bn-something-something-v2/frames'
    # target_path = '/data/xxxx/work/dataset'
    # version = 'V2'
    # txt_file = 'v2-train'
    # process_data(source_path, target_path, version, txt_file)
    DEMO_DIR = '/data/20bn-something-something-v2/frames/181804'
    OUT_DIR = '/data/tmp_data/detectron_vis/RPN-sth-output'
    task_demo(DEMO_DIR, OUT_DIR, 0.8)
