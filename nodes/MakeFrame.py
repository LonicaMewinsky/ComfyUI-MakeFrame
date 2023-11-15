import cv2
import torch
import numpy as np
import os
import random
from PIL import Image
from .. import makeframeutils as mfu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BreakFrames:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_input": ("STRING", {
                    "multiline": False,
                    "default": "C:/Videos/video.mp4"
                }),
            },
        }
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Frames",)

    FUNCTION = "breakframes"
    CATEGORY = "Frames"

    def breakframes(self, file_input):
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"File '{file_input} cannot be found.'")
        # Open the video file
        try:
            video_capture = cv2.VideoCapture(file_input)
            # Check if the anim was loaded
            if not video_capture.isOpened():
                print(f"Error: Could not open video file {file_input}.")
                return None
        except:
            print(f"Error: Could not open video file {file_input}.")
            return None
        # Read each frame from anim
        tensors = []
        while True:
            ret, frame = video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensors.append(mfu.pil_to_tens(frame).to(device))
            else:
                break
        video_capture.release()
        cat_frame_tensors = torch.cat(tensors, dim = 0).to(device)

        return (cat_frame_tensors,)

class GetKeyFrames:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", ),
                "num_keyframes": ("INT", {
                        "default": 12, 
                        "min": 2,
                        "max": 4096,
                        "step": 1
                }),
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Keyframes", "Labeled Keyframes")

    FUNCTION = "getkeyframes"
    CATEGORY = "Frames"

    def getkeyframes(self, frames, num_keyframes):
        N = np.clip(num_keyframes, 2, len(frames)-2)
        frames = frames
        differences = [torch.norm(frames[i+1] - frames[i], p=2) for i in range(len(frames)-1)]
        _, top_indices = torch.topk(torch.tensor(differences), k=N, largest=True)
        keyframe_indices = sorted([index.item() + 1 for index in top_indices])
        keyframe_indices.insert(0, 0)
        keyframe_indices.append(len(frames)-1)
        cat_keyframe_tensors = [frames[i].to(device).unsqueeze(0) for i in keyframe_indices]
        cat_keyframe_tensors = torch.cat(cat_keyframe_tensors, dim = 0).to(device)

        pils = mfu.cat_to_pils(cat_keyframe_tensors)
        keyframes_labeled = [mfu.ImgLabeler(frame, str(idx), size=72, color="#ffffff") for frame, idx in zip(pils, keyframe_indices)]
        keyframe_labeled_tensors = [mfu.pil_to_tens(keyframe) for keyframe in keyframes_labeled]
        cat_keyframe_tensors_labeled = torch.cat(keyframe_labeled_tensors, dim = 0).to(device)
        
        return (cat_keyframe_tensors, cat_keyframe_tensors_labeled)

class MakeGrid:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", ),
                "max_width": ("INT", {
                        "default": 2048, 
                        "min": 64,
                        "max": 8000,
                        "step": 8
                }),
                "max_height": ("INT", {
                        "default": 2048, 
                        "min": 64,
                        "max": 8000,
                        "step": 8
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("Grid", "Rows", "Columns")

    FUNCTION = "makegrid"
    CATEGORY = "Frames"

    def makegrid(self, frames, max_width, max_height):

        pils = mfu.cat_to_pils(frames)
        pils = mfu.normalize_size(pils) #normalize sizes to /8
        rows, cols = mfu.get_grid_aspect(len(pils), pils[0].width, pils[0].height)
        if len(pils) < rows*cols:
            pils = mfu.padlist(pils, rows*cols) #pad list with repeats (black space bad)
        if not rows == 8: max_height = mfu.closest_lcm(max_height, 8, rows)
        if not cols == 8: max_width = mfu.closest_lcm(max_width, 8, cols)

        grid = mfu.constrain_image(mfu.MakeGrid(pils, rows, cols), max_width, max_height)
        grid = mfu.pil_to_tens(grid).to(device)
        return (grid, rows, cols)

class BreakGrid:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid": ("IMAGE",),
                "rows": ("INT",{}),
                "columns": ("INT",{}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Frames",)

    FUNCTION = "breakgrid"
    CATEGORY = "Frames"

    def breakgrid(self, grid, rows, columns):
        pilgrids = mfu.cat_to_pils(grid)
        frames =[]
        for pilgrid in pilgrids:
            frames.extend(mfu.BreakGrid(pilgrid, rows, columns))
        frame_tensors = [mfu.pil_to_tens(frame) for frame in frames]
        cat_frame_tensors = torch.cat(frame_tensors, dim = 0).unsqueeze(0)

        return (cat_frame_tensors)

class RandomImageFromDir:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dir": ("STRING", {
                    "multiline": False,
                    "default": "C:/Poses"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Random Image",)

    FUNCTION = "getrandom"
    CATEGORY = "_for_testing"

    def IS_CHANGED(cls, dir):
        return random.random()
    
    def getrandom(self, dir):
        files = os.listdir(dir)
        image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]

        random_image = random.choice(image_files)

        img = Image.open(os.path.join(dir, random_image))
        tensor = mfu.pil_to_tens(img).unsqueeze(0)
        random_image = None
        return tensor