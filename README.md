# MakeFrame

## BreakFrames:
Break any animation file that OpenCV will process into individual frames.

![image](https://github.com/LonicaMewinsky/ComfyUI-MakeFrame/assets/93007558/2d8e9b0b-cbad-4018-bd02-2d139071738b)

## GetKeyFrames:
Process scene detection on all given frames, rank-orders, returns top N results. Always includes first frame. Also returns labeled images.

![image](https://github.com/LonicaMewinsky/ComfyUI-MakeFrame/assets/93007558/ce9de415-d9c4-43ac-94ba-7b8af52e5927)

## MakeGrid:
Plots given images on a grid. Calculates number of rows and columns for most even aspect ratio.
* For concurrent frame generation. Creates potentially massive image.
* Images are resized to be evenly-divisible by rows, columns, and 8 (to prevent "resize wobble").
* Empty cells are padded with repeats of last image. Black space frustrates generation.

![image](https://github.com/LonicaMewinsky/ComfyUI-MakeFrame/assets/93007558/ac61e777-b5d9-48d5-b20f-57ff1c320d7c)

## BreakGrid:
Breaks given image grid(s) back into individual images. Good practice to use rows/columns output from MakeGrid.

![image](https://github.com/LonicaMewinsky/ComfyUI-MakeFrame/assets/93007558/6a9a5743-4a43-470b-8e10-2f96a2836c8d)
