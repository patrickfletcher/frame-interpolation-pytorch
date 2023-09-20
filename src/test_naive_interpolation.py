import time
import math
import torch
import numpy as np
from PIL import Image
from film.interpolator import Interpolator as FilmInterpolator
from film.util import *
import bisect

FILMModelPath = "./models/film_net_fp16.pt"

# There are many possible interpolation "schedules" with different performance/quality tradeoffs

# different options for display:
# - fixed display rate + fixed frame number --> pauses (variable length if BLIP interferes)
# - continuous mode: generate enough frames so the pauses can be absorbed
# -- maybe could have a control system that varies FPS (slowly/by small amounts) to deal with the BLIP disturbances

# Formulas for bisection-generated frames
# - the interpolated frames from bisection form a complete binary tree, each node is an interpolated frame
# - H = height of tree
# -- number of nodes in the tree:      N = 2**(H+1)-1
# -- number of leaf nodes at height H: L = 2**H = (N+1)/2
# -- number of intervals at height H:  I = N + 1 = 2**(H+1)

# example: two bisections (H=1)
# 1. bisection H=1 --> 3 frames
# A. in each remaining interval (), get 1 averaged frame --> I = 2**(1+1) = 4
# --> total frames = 3 + 4 = 7
# B. in each remaining interval (), get 3 frames --> 3 * I = 3*2**(1+1) = 12
# --> total frames = 3 + 12 = 15

class Config():
    def __init__(self):
        self.num_interp = 31
        self.doSecondaryInterpolation = False
        self.targetFps = 10
        self.save_path = "./output/output.mp4"

config = Config()

def findBisectionLevel(total):
    H=0
    while 2**(H+1 + 1)-1 < total:
        H = H+1

    N = 2**(H+1)-1
    return H, N

class FrameGenerator():

    def __init__(self):
        self.FILMModel = FilmInterpolator()
        self.FILMModel.load_state_dict(torch.load(FILMModelPath))
        self.FILMModel.eval().half()
        self.FILMModel.to(device=torch.device('cuda'))

    def interpolate(self, prevImage, newImage):
        device = torch.device('cuda')
        precision = torch.float16
        tStart = time.time()

        prevImage = np.array(prevImage)
        img0 = prevImage.astype(np.float32) / np.float32(255)
        img0, crop0 = pad_batch(np.expand_dims(img0, axis=0), 64)

        newImage = np.array(newImage)
        img1 = newImage.astype(np.float32) / np.float32(255)
        img1, crop1 = pad_batch(np.expand_dims(img1, axis=0), 64)
    
        img0 = torch.from_numpy(img0).permute(0, 3, 1, 2).half().to(device)
        img1 = torch.from_numpy(img1).permute(0, 3, 1, 2).half().to(device)


        results = [img0, img1]
        idxes = [0, config.num_interp + 1]
        remains = list(range(1, config.num_interp + 1))
        splits = torch.linspace(0, 1, config.num_interp + 2)

        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            # dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])
            dt = (splits[remains[step]] - splits[idxes[start_i]]) / (splits[idxes[end_i]] - splits[idxes[start_i]])
            dt = [dt.tolist()]
            with torch.no_grad():
                prediction = self.FILMModel(x0, x1, 0, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction[0].clamp(0, 1))
            del remains[step]
            
            # results = self.FILMModel(img0, img1, 0, [0.5])

        # add an additional frame between each exisiting frame by averaging successive pairs
        if config.doSecondaryInterpolation:
            results = [img0] + results + [img1]
            nIntervals = len(results)-1
            for ii in range(nIntervals):
                i = 2*ii
                j = 2*ii+1
                xa = results[i]
                xb = results[i+1]
                xc = torch.mean(torch.stack((xa,xb),dim=0),dim=0)
                results.insert(j, xc)
            del results[0]
            del results[-1]

        y1, x1, y2, x2 = crop0
        # frames = [(tensor[0].clamp(0.0, 1.0).cpu() * 255).to(torch.uint8).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
        
        print(torch.cuda.max_memory_allocated())

        ###################################################
        # for opencv saving:
        # append the originals for making the movie
        # results = [img0] + results + [img1]
        frames = [(tensor[0].clamp(0.0, 1.0).cpu() * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

        tnow = time.time()
        print(f"SD time --- FILM interpolation : {tnow-tStart:0.4} s ({len(results)-2} steps)")

        for i, frame in enumerate(frames):
            cv2.imwrite(f"./output/out{i:0d}.png", frame)

        w, h = frames[0].shape[1::-1]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(config.save_path, fourcc, config.targetFps, (w, h))
        # forward
        for frame in frames:
            writer.write(frame)
        # reverse
        for frame in frames[::-1]:
            writer.write(frame)

        writer.release()


if __name__=="__main__":
    fg = FrameGenerator()
    img0 = cv2.cvtColor(cv2.imread("./photos/one.png"), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(cv2.imread("./photos/two.png"), cv2.COLOR_BGR2RGB)

    print(img0.shape)

    fg.interpolate(img0, img1)