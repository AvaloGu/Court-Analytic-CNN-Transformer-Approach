import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
import random
from itertools import groupby
from abc import ABC, abstractmethod
from Loaders.vocab import STOISTAGE1, STOISTAGE2, POINTSEPARATOR


class DataLoader:
    def __init__(self, process_rank = 0, num_processes = 1, mode = "train", infer_folder = None):
        """ mode has to be either train, val, or test, infer
        """
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.mode = mode

        self.not_empty = True # this indicates whether there are still pts in the loader during eval

        if self.mode == "train":
            self.imgfolder = "Dataset/frames"
        elif self.mode == "val":
            self.imgfolder = "Dataset/val_set"
        elif self.mode == "test":
            self.imgfolder = "Dataset/test_set"
        else: # inference
            self.imgfolder = infer_folder

        self.files = sorted([f for f in os.listdir(self.imgfolder) if f.endswith(".jpg")])
        print(f"{len(self.files)} images from {self.imgfolder}")
        
        if self.mode == "train":
            self.frame_target = np.loadtxt("Dataset/TrainingStage1_label.csv", delimiter=",", dtype=str)
        elif self.mode == "val":
            self.frame_target = np.loadtxt("Dataset/ValStage1_label.csv", delimiter=",", dtype=str)
        elif self.mode == "test":
            self.frame_target = np.loadtxt("Dataset/TestStage1_label.csv", delimiter=",", dtype=str)
        else: # inference
            self.frame_target = None

        if self.frame_target is not None:
            assert len(self.files) == len(self.frame_target), "number of images and targets must match"

    def _transform(self, clip_files):
        if self.mode == "train":
            img_process = transforms.Compose([transforms.RandomResizedCrop(224, scale=[0.85, 1], ratio=[0.75, 1.25]), 
                                            # transforms.RandomInvert(p=0.5),
                                            # transforms.RandomGrayscale(p=0.5),
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])
        else:
            img_process = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                std=[0.229, 0.224, 0.225])])
        
        seed = torch.randint(0, 2**32, (1,)).item()
        frames = []
        for f in clip_files:
            img = Image.open(os.path.join(self.imgfolder, f)).convert("RGB")
            # reset the global random number generator (RNG) back to 'seed' every iter
            # this makes sure the same transformation is applied to every frame in the clip
            torch.manual_seed(seed) 
            frame = img_process(img)
            frames.append(frame)

        return torch.stack(frames, dim=0) # (B, 3, 224, 224)

    @abstractmethod
    def next_batch(self):
        pass



class DataLoaderStage0(DataLoader):
    def __init__(self, process_rank = 0, num_processes = 1, mode = "train", infer_folder = None):
        # this loader only supports training, no inferencing

        super().__init__(process_rank, num_processes, mode, infer_folder)

        self.B = 128

        self.indices = list(range(len(self.files)))

        if self.mode == "train":
            random.shuffle(self.indices) # completely randomized batch

        self.current_position = self.process_rank # process_rank starts at 0

    def next_batch(self):
        indices = self.indices[self.current_position : self.current_position + self.B]
        clip_files = [self.files[i] for i in indices]

        x = self._transform(clip_files) # (B, 3, 224, 224)

        y = [STOISTAGE1[s] for s in self.frame_target[indices]] 
        y = torch.tensor(y) # (B,)

        # advance the current position
        self.current_position += self.num_processes * self.B

        # if loading the next batch would be out of bounds, reset
        if self.current_position >= len(self.indices):
            self.current_position = self.process_rank
            random.shuffle(self.indices)
            self.not_empty = False # for eval
        return x, y
                                                                


class DataLoaderStage1(DataLoader):
    def __init__(self, process_rank = 0, num_processes = 1, mode = "train", infer_folder = None):
        super().__init__(process_rank, num_processes, mode, infer_folder)

        self.B = 152
        self.batch_sizes = [16, 32, 48, 64, 96, 128, 152]
        
        self.indices_original = list(range(len(self.files)))
        self.indices = self.indices_original.copy()

    def next_batch(self):
        # this is inefficient dataloader due to deleting elements from a dynamic array (python list)
        # but we do want the randomness in both batch size and the segment we are fetching from the video
        # this helps in both convergence and transfering to stage 2.
        # DDP won't work

        if self.mode == "train":
            random_index = random.randrange(len(self.indices) - 15)
            random_batch_size = random.choice(self.batch_sizes)
            self.B = random_batch_size
        else: # during eval or infer, no randomization
            random_index = 0
            random_batch_size = self.B 

        # python handle out of bound slicing gracefully
        indices = self.indices[random_index: random_index + random_batch_size]
        del self.indices[random_index: random_index + random_batch_size]
        clip_files = [self.files[i] for i in indices] # self.files is list not numpy array

        x = self._transform(clip_files) # (B, 3, 224, 224)

        y = None
        if self.frame_target is not None: # not inference
            y = [STOISTAGE1[s] for s in self.frame_target[indices]] 
            y = torch.tensor(y) # (B,)

        if len(self.indices) < 16:
            if self.mode == "train":
                self.indices = self.indices_original.copy()
            else:
                if len(self.indices) == 0:
                    self.not_empty = False

        return x, y
    



class DataLoaderStage2(DataLoader):
    def __init__(self, process_rank = 0, num_processes = 1, mode = "train", infer_folder = None, stage1_labels = None):
        super().__init__(process_rank, num_processes, mode, infer_folder)

        self.num_examples = 0
        
        if self.mode == "train":
            target = np.loadtxt("Dataset/TrainingStage2_label.csv", delimiter=",", dtype=str)
        elif self.mode == "val":
            target = np.loadtxt("Dataset/ValStage2_label.csv", delimiter=",", dtype=str)
        elif self.mode == "test":
            target = np.loadtxt("Dataset/TestStage2_label.csv", delimiter=",", dtype=str)
        else: # infer
            target = None

        if target is not None:
            encode = [STOISTAGE2[s] for s in target]
            self.target = [] # a list of lists
            # the point separator 'z' is encoded as 31, vocab_size is 32
            for k, g in groupby(encode, lambda x: x==POINTSEPARATOR):
                if not k:
                    self.target.append(list(g))
        else: # during inferece
            self.target = None

        if stage1_labels is not None: # during inference
            frame_label = stage1_labels
        else:
            frame_label = [STOISTAGE1[s] for s in self.frame_target]
            frame_label = torch.tensor(frame_label)

        # we only care about whether a frame is during point or out of point here
        for i in range(2, 6):
            frame_label[frame_label == i] = 1 # we'll set anything besides during_pt as out_of_pt
        label_original = frame_label[:-1]
        label_shift_by_1 = frame_label[1:]
        diff = label_shift_by_1 - label_original
        in_point_out_point_indices = (torch.where((diff==1) | (diff==-1)))[0]

        # indices of the in point frames
        self.indices = [] # a list of lists, each sublist is a point
        for a, b in zip(in_point_out_point_indices[:-1:2], in_point_out_point_indices[1::2]):
            self.indices.append(list(range(a.item() - 4, b.item() + 4))) # we add a bit more buffer, 4 frames

        # TODO: you might want to add a preprocessing step to check for this condition
        assert len(self.indices) == len(self.target), "number of points must match number of points in target"

        if self.mode == "train":
            self._shuffle_points()

        self.current_position = self.process_rank # process_rank starts at 0

    def _shuffle_points(self):
        # shuffle but maintain the relation between labels and frame indices
        label_frames_combined = list(zip(self.target, self.indices))
        random.shuffle(label_frames_combined)
        shuffled_labels, shuffled_frame_indices = zip(*label_frames_combined) # tuples
        self.target = list(shuffled_labels)
        self.indices = list(shuffled_frame_indices)

    def next_batch(self):
        indices = self.indices[self.current_position]
        clip_files = [self.files[i] for i in indices]

        x_frames = self._transform(clip_files) # (B, 3, 224, 224)

        y = None
        if self.target is not None: # not during inference
            y = self.target[self.current_position]
            self.num_examples = len(y)
            y = torch.tensor(y) # (Num of shots in this point,)
            # add point token to the end
            point_token = torch.tensor([POINTSEPARATOR], dtype=torch.long, device=y.device)
            y = torch.cat((point_token, y, point_token), dim=0) # (Num of shots in this point+2,)
        
        # advance the current position
        self.current_position += self.num_processes

        # if loading the next batch would be out of bounds, reset
        if self.current_position >= len(self.target):
            if self.mode == "train":
                self._shuffle_points()
                self.current_position = self.process_rank
            else:
                self.not_empty = False

        return x_frames, y # (B, 3, 224, 224), (Num of shots in this point+2,)