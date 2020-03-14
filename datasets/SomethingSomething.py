import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import copy
import json
import numpy as np

from utils import load_value_file


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


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            assert False

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(root_path, annotation_path, proposal_path, subset, n_samples_for_each_video):
    list_file = os.path.join(annotation_path, subset)
    data = [VideoRecord(x.strip().split(' ')) for x in open(list_file)]
    proposal_file = os.path.join(proposal_path, subset.replace('txt', 'json'))
    box = json.load(open(proposal_file))

    dataset = []
    for i in range(len(data)):
        if i % 10000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(data)))

        video_path = os.path.join(root_path, data[i].path)
        proposal = box[data[i].path]

        n_frames = data[i].num_frames
        if n_frames <= 0:
            assert False, 'Number of frames cannot be negetive.'

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'proposal': proposal,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': data[i].path,
            'label': data[i].label
        }

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                for j in range(n_samples_for_each_video):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list(range(1, n_frames + 1))
                    dataset.append(sample_j)
            else:
                assert False

    return dataset


class STHV1(data.Dataset):
    def __init__(self,
                 root_path, 
                 annotation_path,
                 proposal_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 n_box_per_frame=10,
                 get_loader=get_default_video_loader):
        self.data = make_dataset(
            root_path, annotation_path, proposal_path, subset, n_samples_for_each_video)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.n_box_per_frame = n_box_per_frame
        self.loader = get_loader()

    def __getitem__(self, index):
        path = self.data[index]['video']
        proposal = self.data[index]['proposal']
        frame_indices = self.data[index]['frame_indices']
        
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        proposal = [proposal[i-1] for i in frame_indices]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            new_clip = []
            new_proposal = []
            for i in range(len(frame_indices)):
                img = clip[i]
                bx = np.array(proposal[i])[:self.n_box_per_frame,:]
                new_img, new_bx = self.spatial_transform(img, bx, self.data[index]['label'])
                new_clip.append(new_img)
                new_proposal.append(new_bx)
            clip = new_clip
            proposal = new_proposal
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        proposal = torch.stack(proposal, 0)
        proposal[:,:,0::2] = proposal[:,:,0::2] * clip.shape[3]
        proposal[:,:,1::2] = proposal[:,:,1::2] * clip.shape[2]
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, proposal, target

    def __len__(self):
        return len(self.data)

class STHV2(data.Dataset):
    def __init__(self,
                 root_path,
                 annotation_path,
                 proposal_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 n_box_per_frame=10,
                 get_loader=get_default_video_loader):
        self.data = make_dataset(
            root_path, annotation_path, proposal_path, subset, n_samples_for_each_video)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.n_box_per_frame = n_box_per_frame
        self.loader = get_loader()

    def __getitem__(self, index):
        path = self.data[index]['video']
        proposal = self.data[index]['proposal']
        frame_indices = self.data[index]['frame_indices']
        
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        proposal = [proposal[i-1] for i in frame_indices]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            new_clip = []
            new_proposal = []
            for i in range(len(frame_indices)):
                img = clip[i]
                bx = np.array(proposal[i])[:self.n_box_per_frame,:]
                new_img, new_bx = self.spatial_transform(img, bx, self.data[index]['label'])
                new_clip.append(new_img)
                new_proposal.append(new_bx)
            clip = new_clip
            proposal = new_proposal
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        proposal = torch.stack(proposal, 0)
        proposal[:,:,0::2] = proposal[:,:,0::2] * clip.shape[3]
        proposal[:,:,1::2] = proposal[:,:,1::2] * clip.shape[2]
        
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, proposal, target

    def __len__(self):
        return len(self.data)