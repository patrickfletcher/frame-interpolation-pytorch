"""The film_net frame interpolator main model code.

Basics
======
The film_net is an end-to-end learned neural frame interpolator implemented as
a PyTorch model. It has the following inputs and outputs:

Inputs:
  x0: image A.
  x1: image B.
  time: desired sub-frame time.

Outputs:
  image: the predicted in-between image at the chosen time in range [0, 1].

Additional outputs include forward and backward warped image pyramids, flow
pyramids, etc., that can be visualized for debugging and analysis.

Note that many training sets only contain triplets with ground truth at
time=0.5. If a model has been trained with such training set, it will only work
well for synthesizing frames at time=0.5. Such models can only generate more
in-between frames using recursion.

Architecture
============
The inference consists of three main stages: 1) feature extraction 2) warping
3) fusion. On high-level, the architecture has similarities to Context-aware
Synthesis for Video Frame Interpolation [1], but the exact architecture is
closer to Multi-view Image Fusion [2] with some modifications for the frame
interpolation use-case.

Feature extraction stage employs the cascaded multi-scale architecture described
in [2]. The advantage of this architecture is that coarse level flow prediction
can be learned from finer resolution image samples. This is especially useful
to avoid overfitting with moderately sized datasets.

The warping stage uses a residual flow prediction idea that is similar to
PWC-Net [3], Multi-view Image Fusion [2] and many others.

The fusion stage is similar to U-Net's decoder where the skip connections are
connected to warped image and feature pyramids. This is described in [2].

Implementation Conventions
====================
Pyramids
--------
Throughtout the model, all image and feature pyramids are stored as python lists
with finest level first followed by downscaled versions obtained by successively
halving the resolution. The depths of all pyramids are determined by
options.pyramid_levels. The only exception to this is internal to the feature
extractor, where smaller feature pyramids are temporarily constructed with depth
options.sub_levels.

Color ranges & gamma
--------------------
The model code makes no assumptions on whether the images are in gamma or
linearized space or what is the range of RGB color values. So a model can be
trained with different choices. This does not mean that all the choices lead to
similar results. In practice the model has been proven to work well with RGB
scale = [0,1] with gamma-space images (i.e. not linearized).

[1] Context-aware Synthesis for Video Frame Interpolation, Niklaus and Liu, 2018
[2] Multi-view Image Fusion, Trinidad et al, 2019
[3] PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
"""
from typing import Dict, List

import torch
from torch import nn

from . import util
from .feature_extractor import FeatureExtractor
from .fusion import Fusion
from .pyramid_flow_estimator import PyramidFlowEstimator


class Interpolator(nn.Module):
    def __init__(
            self,
            pyramid_levels=7,
            fusion_pyramid_levels=5,
            specialized_levels=3,
            sub_levels=4,
            filters=64,
            flow_convs=(3, 3, 3, 3),
            flow_filters=(32, 64, 128, 256),
    ):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels

        self.extract = FeatureExtractor(3, filters, sub_levels)
        self.predict_flow = PyramidFlowEstimator(filters, flow_convs, flow_filters)
        self.fuse = Fusion(sub_levels, specialized_levels, filters)

    def flow_and_fuse(self, image_pyramids, feature_pyramids, time_steps):
        # Predict forward flow.
        forward_residual_flow_pyramid = self.predict_flow(feature_pyramids[0], feature_pyramids[1])

        # Predict backward flow.
        backward_residual_flow_pyramid = self.predict_flow(feature_pyramids[1], feature_pyramids[0])

        # Concatenate features and images:

        # Note that we keep up to 'fusion_pyramid_levels' levels as only those
        # are used by the fusion module.

        forward_flow_pyramid = util.flow_pyramid_synthesis(forward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        backward_flow_pyramid = util.flow_pyramid_synthesis(backward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        pyramids_to_warp = [
            util.concatenate_pyramids(image_pyramids[0][:self.fusion_pyramid_levels],
                                      feature_pyramids[0][:self.fusion_pyramid_levels]),
            util.concatenate_pyramids(image_pyramids[1][:self.fusion_pyramid_levels],
                                      feature_pyramids[1][:self.fusion_pyramid_levels])
        ]

        results = []
        for time_step in time_steps:
            mid_time = torch.full((1,1), time_step, dtype=image_pyramids[0][0].dtype, device=image_pyramids[0][0].device)
            backward_flow = util.multiply_pyramid(backward_flow_pyramid, mid_time[:, 0])
            forward_flow = util.multiply_pyramid(forward_flow_pyramid, 1 - mid_time[:, 0])

            # Warp features and images using the flow. Note that we use backward warping
            # and backward flow is used to read from image 0 and forward flow from
            # image 1.
            forward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[0], backward_flow)
            backward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[1], forward_flow)

            aligned_pyramid = util.concatenate_pyramids(forward_warped_pyramid,
                                                        backward_warped_pyramid)
            aligned_pyramid = util.concatenate_pyramids(aligned_pyramid, backward_flow)
            aligned_pyramid = util.concatenate_pyramids(aligned_pyramid, forward_flow)
            results.append(self.fuse(aligned_pyramid))

        return results
     
    def recursively_bisect(self, image_pyramids, feature_pyramids, bisections_remaining, final_timesteps):
        if(bisections_remaining == 0):
            # for the final layer of the bisections, use final_timesteps (to squeeze out some extra frames
            # using multiple timesteps: even the model doesn't work as well outside of t=0.5, changes should be
            # small here that it may not matter much)
            return self.flow_and_fuse(image_pyramids, feature_pyramids, final_timesteps)

        # for all other layers, time_step is 0.5
        this_midpoint_image = self.flow_and_fuse(image_pyramids, feature_pyramids, [0.5])[0]
        
        # bisect left
        left_image_pyramids = [image_pyramids[0], util.build_image_pyramid(this_midpoint_image, self.pyramid_levels)]
        left_feature_pyramids = [feature_pyramids[0], self.extract(left_image_pyramids[1])]
        left_result = self.recursively_bisect(left_image_pyramids, left_feature_pyramids, bisections_remaining-1, final_timesteps)

        # free up the LHS memory right away
        image_pyramids[0] = None
        feature_pyramids[0] = None
        left_image_pyramids[0] = None
        left_feature_pyramids[0] = None

        # bisect right 
        right_image_pyramids = [left_image_pyramids[1], image_pyramids[1]]
        right_feature_pyramids = [left_feature_pyramids[1], feature_pyramids[1]]
        right_result = self.recursively_bisect(right_image_pyramids, right_feature_pyramids, bisections_remaining-1, final_timesteps)
        return left_result + [this_midpoint_image] + right_result

    @torch.jit.export
    def forward(self, x0, x1, num_bisections, final_timesteps) -> List[torch.Tensor]:
        image_pyramids =  [util.build_image_pyramid(x0, self.pyramid_levels), util.build_image_pyramid(x1, self.pyramid_levels)]
        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]
        return self.recursively_bisect(image_pyramids, feature_pyramids, num_bisections, final_timesteps)
