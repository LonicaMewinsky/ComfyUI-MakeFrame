import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import os
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
        frame_count = 0
        transformer = transforms.ToTensor()
        tensors = []
        while True:
            ret, frame = video_capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensors.append(transformer(frame).to(device).unsqueeze(0))
                frame_count += 1
            else:
                break
        video_capture.release()
        cat_frame_tensors = torch.cat(tensors, dim = 0).permute(0, 2, 3, 1)

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
    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("Keyframes", "Keyframe indices",)

    FUNCTION = "getkeyframes"
    CATEGORY = "Frames"

    def getkeyframes(self, frames, num_keyframes):
        N = np.clip(num_keyframes, 2, len(frames)-1) # Save a spot for first frame
        frames = frames
        differences = [torch.norm(frames[i+1] - frames[i], p=2) for i in range(len(frames)-1)]
        _, top_indices = torch.topk(torch.tensor(differences), k=N, largest=True)
        keyframe_indices = sorted([index.item() + 1 for index in top_indices])
        keyframe_indices.insert(0, 0)
        cat_keyframe_tensors = [frames[i].to(device).unsqueeze(0) for i in keyframe_indices]
        cat_keyframe_tensors = torch.cat(cat_keyframe_tensors, dim = 0)
        return (cat_keyframe_tensors, keyframe_indices)

class MakeGrid:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", ),
                "grid_rows": ("INT", {
                        "default": 4, 
                        "min": 2,
                        "max": 24,
                        "step": 1
                }),
                "grid_cols": ("INT", {
                        "default": 4, 
                        "min": 2,
                        "max": 24,
                        "step": 1
                }),
                "max_width": ("INT", {
                        "default": 1024, 
                        "min": 64,
                        "max": 4096,
                        "step": 8
                }),
                "max_height": ("INT", {
                        "default": 1024, 
                        "min": 64,
                        "max": 4096,
                        "step": 8
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Grid",)

    FUNCTION = "makegrid"
    CATEGORY = "Frames"

    def makegrid(self, frames, grid_rows, grid_cols, max_width, max_height):
        # Pad the list with extras; black space frustrates generation

        # Size needs to be divisible by 8 AND the grid dimension
        if not grid_rows == 8: max_height = mfu.closest_lcm(max_height, 8, grid_rows)
        if not grid_cols == 8: max_width = mfu.closest_lcm(max_width, 8, grid_cols)
        # Build base grid
        pils = mfu.cat_to_pils(frames)
        pils = mfu.normalize_size(pils) #normalize sizes to /8
        grid = mfu.constrain_image(mfu.MakeGrid(pils, grid_rows, grid_cols), max_width, max_height)
        grid = mfu.pil_to_cat(grid)

        return (grid,)