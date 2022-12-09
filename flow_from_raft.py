import os
import sys
import numpy as np
import av
import glob
from natsort import natsorted, ns


#import pickle
#import seaborn as sns

import cv2

from matplotlib import pyplot as plt

import logging

logger = logging.getLogger(__name__)
# These lines allow me to see logging.info messages in my jupyter cell output
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)


class flow_source():

    def __init__(self, file_path, out_parent_dir="frames_out"):

        self.file_path = file_path
        self.source_file_name = os.path.split(file_path)[-1].split('.')[0]
        self.source_file_suffix = os.path.split(file_path)[-1].split('.')[1]

        self.out_parent_dir = out_parent_dir
        self.raw_frames_path = os.path.join(out_parent_dir, self.source_file_name, 'raw')
        self.video_out_path = os.path.join(out_parent_dir, self.source_file_name, 'videos')


    def write_video_from_frames(self, frames_dir, video_name, fps = 24):

        logger.info("Writing movie...")

        from PIL import Image

        if os.path.isdir(self.video_out_path) is False:
            os.makedirs(self.video_out_path)

        images = glob.glob(os.path.join(frames_dir, '*.png')) + \
                 glob.glob(os.path.join(frames_dir, '*.jpg'))

        images = natsorted(images, key=lambda y: y.lower())  # sort alphanumeric in ascending order
        #images = sorted(images)

        total_frames = len(images)


        container = av.open(os.path.join(self.video_out_path,video_name), mode="w")

        img = Image.open(images[0])
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = img.width
        stream.height = img.height
        stream.pix_fmt = "yuv420p"

        #for frame_i in range(total_frames):
        for image_path in images:

            frame = av.VideoFrame.from_image(Image.open(image_path))

            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()

        logger.info("Done writing movie.")

    def write_video_from_raft(self):

        flow_frames_path = os.path.join(source.out_parent_dir, source.source_file_name, 'raft_frames')
        self.write_video_from_frames(flow_frames_path, 'raft_flow.mp4')

    def extract_frames(self):

        logger.info("Extracting frames...")

        if os.path.isdir(self.raw_frames_path) is False:
            os.makedirs(self.raw_frames_path)

        container = av.open(self.file_path)

        # take first video stream
        stream = container.streams.video[0]

        # get video fps
        average_fps = int(stream.average_rate)
        for frame in container.decode(video=0):
            frame.to_image().save(os.path.join(self.raw_frames_path,'frame-{}.png'.format(frame.index)))

        logger.info("Done extracting frames.")

    def estimate_flow_with_raft(self):

        logger.info("Estimating flow from raft...")

        import torch
        import argparse

        from PIL import Image

        sys.path.append('core')
        from raft import RAFT
        from utils.utils import InputPadder
        from utils import flow_viz
        # import matplotlib.pyplot as plt

        flow_frames_path = os.path.join(self.out_parent_dir, self.source_file_name, 'raft_frames')
        if os.path.isdir(flow_frames_path) is False:
            os.makedirs(flow_frames_path)

        def load_image(imfile):
            img = np.array(Image.open(imfile)).astype(np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img[None].to('cuda')


        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.path = self.raw_frames_path
        args.alternate_corr = False
        args.mixed_precision = False
        args.model = "models/raft-kitti.pth"
        args.small = False
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))

        model = model.module
        model.to('cuda')
        model.eval()

        with torch.no_grad():

            images = glob.glob(os.path.join(args.path, '*.png')) + \
                     glob.glob(os.path.join(args.path, '*.jpg'))

            images = natsorted(images, key=lambda y: y.lower())  # sort alphanumeric in ascending order

            index = 0
            for imfile1, imfile2 in zip(images[:-1], images[1:]):

                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                #viz(image1, flow_up)

                img = image1[0].permute(1, 2, 0).cpu().numpy()
                flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

                # map flow to rgb image
                flo = flow_viz.flow_to_image(flo)
                img_flo = np.concatenate([img, flo], axis=0)
                cv2.imwrite(str(os.path.join(flow_frames_path,'frame-{}.png'.format(index))),img_flo)
                index += 1

            logger.info("Done estimating flow from raft")

if __name__ == "__main__":

    a_file_path = os.path.join("videos/", "optic_flow_snippet.mp4")
    source = flow_source(a_file_path)
    source.extract_frames()
    source.estimate_flow_with_raft()
    source.write_video_from_raft()


