import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import os

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
                "num_keyframes": ("INT", {
                        "default": 0, 
                        "min": 0,
                        "max": 4096,
                        "step": 1
                }),
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("Frames","Keyframes")

    FUNCTION = "breakframes"
    CATEGORY = "Frames"

    def breakframes(self, file_input, num_keyframes):
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
                tensors.append(transformer(frame).unsqueeze(0))
                frame_count += 1
            else:
                break
        video_capture.release()
        if num_keyframes > 0:
            N = np.clip(num_keyframes, 2, len(tensors)-1) # Save a spot for first frame
            differences = [torch.norm(tensors[i+1] - tensors[i], p=2) for i in range(len(tensors)-1)]
            _, top_indices = torch.topk(torch.tensor(differences), k=N, largest=True)
            keyframe_indices = sorted([index.item() + 1 for index in top_indices])
            keyframe_indices.insert(0, 0)
            cat_keyframe_tensors = [tensors[i] for i in keyframe_indices]
            cat_keyframe_tensors = torch.cat(cat_keyframe_tensors, dim = 0).permute(0, 2, 3, 1)
        else:
            cat_keyframe_tensors = None
        cat_frame_tensors = torch.cat(tensors, dim = 0).permute(0, 2, 3, 1)
        return (cat_frame_tensors, cat_keyframe_tensors)

NODE_CLASS_MAPPINGS = {
    "BreakFrames": BreakFrames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BreakFrames": "BreakFrames"
}