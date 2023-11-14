import os
import sys
import subprocess

here = os.path.dirname(__file__)
requirements_path = os.path.join(here, "requirements.txt")

try:
    from .nodes.MakeFrame import BreakFrames, GetKeyFrames, MakeGrid, BreakGrid
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
    from .nodes.MakeFrame import BreakFrames, GetKeyFrames, MakeGrid, BreakGrid

NODE_CLASS_MAPPINGS = {
    "BreakFrames": BreakFrames,
    "GetKeyFrames": GetKeyFrames,
    "MakeGrid": MakeGrid,
    "BreakGrid": BreakGrid,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BreakFrames": "BreakFrames",
    "GetKeyFrames": "GetKeyFrames",
    "MakeGrid": "MakeGrid",
    "BreakGrid": "BreakGrid",
}