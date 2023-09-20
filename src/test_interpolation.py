import time
import math
import torch
import numpy as np
from PIL import Image
from film.interpolator import Interpolator as FilmInterpolator
from film.util import *

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
        self.sdWorkerFixedCycleTime = 2.5
        self.targetFps = 10
        self.bisectionLevels = 3
        self.doSecondaryInterpolation = False
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

        total_frames_needed = math.ceil(config.sdWorkerFixedCycleTime * config.targetFps)
        # already have the one new frame from SD
        num_interpolated_frames = total_frames_needed - 1

        # we'll get half the new frames from secondary interpolation if
        # that's enabled (one less that the total frames because we
        if config.doSecondaryInterpolation:
            num_interpolated_frames = math.ceil(num_interpolated_frames / 2)

        print(f" total_frames_needed: {total_frames_needed}\n num_interpolated_frames: {num_interpolated_frames}\n")

        num_bisections, num_bisection_frames = findBisectionLevel(num_interpolated_frames) 
        print(f" num_bisections: {num_bisections}\n num_bisection_frames: {num_bisection_frames}\n bisection + secondary: {2**(num_bisections+2)-1}\n")

        with torch.no_grad():
            # With multiple time steps, FILM can get a nice sharp flow (and
            # runs much faster per frame, as it just need to redo the
            # fusion layer), but it has trouble at the edges to actually line
            # up to the start/end images.
            # To address this, use bisection for 30% of the frames at
            # the start and end (15% each) to connect the main chunk
            # to the start/end frames
            bisections_per_side = max(
                0.0,
                round(math.log(num_interpolated_frames*0.15, 2)-1)
            )
            bisection_total_frames = int(2*(2**(bisections_per_side+1)-1))
            midsection_total_frames = int(num_interpolated_frames - bisection_total_frames)
            print(f" bisection_total_frames: {bisection_total_frames}\n midsection_total_frames: {midsection_total_frames}\n")
            bisection_pct = bisection_total_frames / num_interpolated_frames
            mid_timesteps = np.linspace(0.0, 1.0, midsection_total_frames).tolist()
            print(f" mid_timesteps: {mid_timesteps}")
            
            mid_results = self.FILMModel(img0, img1, 0, mid_timesteps)
            start_results = self.FILMModel(img0, mid_results[0], bisections_per_side, [0.5]) # why does this not generate a duplicate, as mid_results[0] was dt=0?
            end_results = self.FILMModel(mid_results[-1], img1, bisections_per_side, [0.5]) 
            results = start_results + mid_results + end_results

            # alternate tests
            #################
            
            # results = self.FILMModel(img0, img1, 3, [0.5])
            # results = self.FILMModel(img0, img1, 2, [0.25, 0.5, 0.75])

            # mid_results = self.FILMModel(img0, img1, num_bisections, [0.5])
            # start_results = self.FILMModel(img0, mid_results[0], 0, [0.5])
            # end_results = self.FILMModel(mid_results[-1], img1, 0, [0.5]) 
            # results = start_results + mid_results + end_results

            # mid_results = self.FILMModel(img0, img1, config.bisectionLevels, [0.5])
            # start_results = self.FILMModel(img0, mid_results[0], 0, [0.5])
            # end_results = self.FILMModel(mid_results[-1], img1, 0, [0.5]) 
            # results = start_results + mid_results + end_results
            

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
        
        ###################################################
        # for opencv saving:
        # append the originals for making the movie
        results = [img0] + results + [img1]
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