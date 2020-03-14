import random
import math


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out
        
        
class TemporalSampling(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        
        step = (len(frame_indices) - 1) / (self.size - 1)
        jittering = step // 2
        out = []
        for i in range(self.size):
            if i > 0 and i < self.size:
                out.append(frame_indices[min(int(step*i+random.randint(0,jittering)), len(frame_indices) - 1)])
            else:
                out.append(frame_indices[min(int(step*i), len(frame_indices) - 1)])

        return out
