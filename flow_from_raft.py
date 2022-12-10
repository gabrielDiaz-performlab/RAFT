import os
import sys
import numpy as np
import av
import glob
import torch

from natsort import natsorted, ns

sys.path.append('core')
from utils.utils import InputPadder
from utils import flow_viz
from PIL import Image

#import pickle
#import seaborn as sns

import cv2 as cv

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

        total_frames = len(images)
        container = av.open(os.path.join(self.video_out_path,video_name), mode="w")

        img = Image.open(images[0])
        stream = container.add_stream("h264", rate=fps)
        stream.width = img.width
        stream.height = img.height

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

    def write_video_from_deepflow(self):

        flow_frames_path = os.path.join(source.out_parent_dir, source.source_file_name, 'deepflow_frames')
        self.write_video_from_frames(flow_frames_path, 'deepflow_flow.mp4')

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
            frame.to_image().save(os.path.join(self.raw_frames_path,'{:06d}.png'.format(frame.index)))

        logger.info("Done extracting frames.")

    def estimate_flow_with_deepflow(self):

        logger.info("Estimating flow with deepflow...")

        flow_frames_path = os.path.join(self.out_parent_dir, self.source_file_name, 'deepflow_frames')

        if os.path.isdir(flow_frames_path) is False:
            os.makedirs(flow_frames_path)

        images = glob.glob(os.path.join(self.raw_frames_path, '*.png')) + \
                 glob.glob(os.path.join(self.raw_frames_path, '*.jpg'))

        df = cv.optflow.createOptFlow_DeepFlow()

        images = natsorted(images, key=lambda y: y.lower())  # sort alphanumeric in ascending order

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        with torch.no_grad():

            index = 0
            for imfile1, imfile2 in zip(images[:-1], images[1:]):

                image1 = cv.imread(imfile1)
                # self.load_image(imfile1)
                image2 = cv.imread(imfile2)

                image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
                image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

                image1_gray = clahe.apply(image1_gray)
                image2_gray = clahe.apply(image2_gray)

                flow = df.calc(image1_gray, image2_gray, flow=None)
                img_flow_as_vectors = self.overlay_flow_as_arrows(image1, flow, 15)

                cv.imwrite(str(os.path.join(flow_frames_path, 'frame-{}.png'.format(index))), img_flow_as_vectors)
                index+=1

        logger.info("Done estimating flow with deepflow.")

    def load_image(self,imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to('cuda')

    def estimate_flow_with_raft(self, viz_type="vector"):

        logger.info("Estimating flow with raft...")

        import argparse
        from raft import RAFT

        flow_frames_path = os.path.join(self.out_parent_dir, self.source_file_name, 'raft_frames')

        if os.path.isdir(flow_frames_path) is False:
            os.makedirs(flow_frames_path)

        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.path = self.raw_frames_path
        args.alternate_corr = False
        args.mixed_precision = False
        args.model = "models/raft-things.pth"
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

                image1 = self.load_image(imfile1)
                image2 = self.load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                img = image1[0].permute(1, 2, 0).cpu().numpy()
                flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

                if viz_type == 'color':

                    # map flow to rgb image
                    img_flow_as_color = flow_viz.flow_to_image(flo)
                    # img_flo = np.concatenate([img, flo], axis=0)
                    cv.imwrite(str(os.path.join(flow_frames_path,'frame-{}.png'.format(index))), img_flow_as_color)

                elif viz_type == 'vector':

                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    img_flow_as_vectors = self.overlay_flow_as_arrows(img,flo,20)

                    cv.imwrite(str(os.path.join(flow_frames_path,'frame-{}.png'.format(index))), img_flow_as_vectors)

                index += 1

            logger.info("Done estimating flow with raft")

    def overlay_flow_as_arrows(self, Image, Flow, Divisor):

        '''Display image with a visualisation of a flow over the top.
        A divisor controls the density of the quiver plot.'''

        def arrowedLine(mask, pt1, pt2, color, thickness):

            line_type = int(8)
            shift = int(0)
            tipLength = float(0.3)
            pi = np.pi

            pt1 = np.array(pt1)
            pt2 = np.array(pt2)

            pt1 = pt1.astype(float)
            pt2 = pt2.astype(float)

            ptsDiff = np.array([pt1[0] - pt2[0], pt1[1] - pt2[1]])
            # Factor to normalize the size of the tip depending on the length of the arrow
            tipSize = cv.norm(ptsDiff) * tipLength

            # Draw main line
            mask = cv.line(mask, (pt1[1].astype(int), pt1[0].astype(int)),
                           (pt2[1].astype(int), pt2[0].astype(int)), color, thickness, line_type, shift)

            # calculate line angle
            angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])

            # draw first line of arrow
            px = round(pt2[0] + tipSize * np.cos(angle + pi / 4))
            py1 = round(pt2[1] + tipSize * np.sin(angle + pi / 4))
            mask = cv.line(mask, (int(py1), int(px)),
                           (int(pt2[1]), int(pt2[0])), color, thickness, line_type, shift)

            # draw second line of arrow
            px = round(pt2[0] + tipSize * np.cos(angle - pi / 4))
            py1 = round(pt2[1] + tipSize * np.sin(angle - pi / 4))
            mask = cv.line(mask, (int(py1), int(px)),
                           (int(pt2[1]), int(pt2[0])), color, thickness, line_type, shift)

            return mask

        PictureShape = np.shape(Image)
        # determine number of quiver points there will be
        Imax = int(PictureShape[0] / Divisor)
        Jmax = int(PictureShape[1] / Divisor)

        # create a blank mask, on which lines will be drawn.
        mask = np.zeros_like(Image)
        for i in range(1, Imax):
            for j in range(1, Jmax):
                X1 = i * Divisor
                Y1 = j * Divisor
                X2 = int(X1 + Flow[X1, Y1, 1])
                Y2 = int(Y1 + Flow[X1, Y1, 0])
                X2 = np.clip(X2, 0, PictureShape[0])
                Y2 = np.clip(Y2, 0, PictureShape[1])
                # add all the lines to the mask
                mask = arrowedLine(mask, (X1, Y1), (X2, Y2), (255, 255, 0), 1)

        # superpose lines onto image
        img = cv.add(Image, mask)
        # plt.imshow(Image)
        # plt.pause(0.05)

        return img



if __name__ == "__main__":
    #a_file_path = os.path.join("videos/", "flowformer.mp4")
    a_file_path = os.path.join("videos/", "optic_flow_snippet.mp4")
    #a_file_path = os.path.join("videos/", "demo_frames.mp4")
    source = flow_source(a_file_path)
    #source.extract_frames()

    #source.estimate_flow_with_raft()
    #source.write_video_from_raft()

    source.estimate_flow_with_deepflow()
    source.write_video_from_deepflow()


