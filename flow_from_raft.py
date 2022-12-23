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

os.add_dll_directory("D://opencvgpu//opencv_build_310//bin")
os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.8//bin")

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

        # -pix_fmt yuv444p

        img = Image.open(images[0])
        stream = container.add_stream("libx264", rate=fps)
        stream.options["crf"] = "0"

        #stream.pix_fmt = "yuv444p"
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

    def write_video_from_nvflow(self):

        flow_frames_path = os.path.join(source.out_parent_dir, source.source_file_name, 'nvflow_frames')
        self.write_video_from_frames(flow_frames_path, 'nvflow_flow.mp4')

    def write_video_from_deepflow(self):

        flow_frames_path = os.path.join(source.out_parent_dir, source.source_file_name, 'deepflow_frames')
        self.write_video_from_frames(flow_frames_path, 'deepflow_flow.mp4')

    def write_video_from_raft(self):

        flow_frames_path = os.path.join(source.out_parent_dir, source.source_file_name, 'raft_frames')
        self.write_video_from_frames(flow_frames_path, 'raft_flow.mp4')

    def write_video_from_brox(self):

        flow_frames_path = os.path.join(source.out_parent_dir, source.source_file_name, 'brox_flow_frames')
        self.write_video_from_frames(flow_frames_path, 'brox_flow.mp4')

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

    def estimate_flow_with_brox(self):

        logger.info("Estimating flow with brox...")

        flow_frames_path = os.path.join(self.out_parent_dir, self.source_file_name, 'brox_flow_frames')

        if os.path.isdir(flow_frames_path) is False:
            os.makedirs(flow_frames_path)

        images = glob.glob(os.path.join(self.raw_frames_path, '*.png')) + \
                 glob.glob(os.path.join(self.raw_frames_path, '*.jpg'))


        images = natsorted(images, key=lambda y: y.lower())  # sort alphanumeric in ascending order

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        index = 0

        # (alpha ,gamma ,scale ,inner_iterations ,outer_iterations ,solver_iterations)
        brox = cv.cuda_BroxOpticalFlow.create(alpha=0.05, gamma = 50.0, scale_factor=1.0)

        # imsize = cv.imread(images[0]).size

        image1_gpu = cv.cuda_GpuMat()
        image2_gpu = cv.cuda_GpuMat()

        # tempImg = cv.cvtColor(cv.imread(images[0]), cv.COLOR_BGR2GRAY)
        # image1_gpu.upload(tempImg)
        # flow = cv.cuda_GpuMat(image1_gpu.size(), cv.CV_32FC2)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):

            image1 = cv.imread(imfile1)
            image2 = cv.imread(imfile2)

            image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY).astype("float32")/255.0
            image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY).astype("float32")/255.0

            # image1_gray = image1.astype("float32") / 255.0
            # image2_gray = image1.astype("float32") / 255.0

            image1_gpu.upload(image1_gray)
            image2_gpu.upload(image2_gray)

            # Image.convertTo(Image, CV_32FC1, 1.0 / 255.0);
            flow = cv.cuda_GpuMat(image1_gpu.size(), cv.CV_32FC2)

            gpu_flow = brox.calc(image1_gpu, image2_gpu, flow)

            flow = gpu_flow.download()

            gpu_flow_x = cv.cuda_GpuMat(gpu_flow.size(), cv.CV_32FC2)
            gpu_flow_y = cv.cuda_GpuMat(gpu_flow.size(), cv.CV_32FC2)
            cv.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y], cv.cuda.Stream_Null())

            gpu_magnitude, gpu_angle = cv.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=False)

            # set value to normalized magnitude from 0 to 1
            gpu_v = cv.cuda.normalize(gpu_magnitude, 0.0, 255.0, cv.NORM_MINMAX, -1)

            res = gpu_v.download()

            img_flow_as_vectors = self.overlay_flow_as_arrows(image1, flow, 15)

            cv.imwrite(str(os.path.join(flow_frames_path, 'frame-{}.png'.format(index))), img_flow_as_vectors)
            index += 1

        logger.info("Done estimating flow with deepflow.")



    def estimate_flow_with_nvflow(self):

        logger.info("Estimating flow with nvflow...")

        flow_frames_path = os.path.join(self.out_parent_dir, self.source_file_name, 'nvflow_frames')

        if os.path.isdir(flow_frames_path) is False:
            os.makedirs(flow_frames_path)

        images = glob.glob(os.path.join(self.raw_frames_path, '*.png')) + \
                 glob.glob(os.path.join(self.raw_frames_path, '*.jpg'))


        images = natsorted(images, key=lambda y: y.lower())  # sort alphanumeric in ascending order

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        index = 0


        image1_gray = cv.cvtColor(cv.imread(images[0]), cv.COLOR_BGR2GRAY)

        nvof = cv.cuda_NvidiaOpticalFlow_2_0.create((image1_gray.shape[1], image1_gray.shape[0]),
                                                    outputGridSize=1,  # 1,2, 4.  Higher is less accurate.
                                                    enableCostBuffer=True,
                                                    enableTemporalHints=True, )

        for imfile1, imfile2 in zip(images[:-1], images[1:]):

            image1 = cv.imread(imfile1)
            image2 = cv.imread(imfile2)

            image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
            image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

            image1_gray = clahe.apply(image1_gray)
            image2_gray = clahe.apply(image2_gray)

            flowTuples = nvof.calc(image1_gray, image2_gray, None)

            # flowOut = cv.cuda_GpuMat((np.shape(flowTuples[0])[1],np.shape(flowTuples[0])[0]), type=cv.CV_32FC3)
            # flowOut = np.zeros(np.shape(flowTuples[0]))

            flow = nvof.convertToFloat(flowTuples[0], np.array(np.shape(flowTuples[0])))
            #flow = nvof.upSampler(flow[0], image1_gray.shape[1], image1_gray.shape[0], nvof.getGridSize(), None)

            # cv2.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)
            nvof.collectGarbage()

            # flow = df.calc(image1_gray, image2_gray, flow=None)
            img_flow_as_vectors = self.overlay_flow_as_arrows(image1, flow, 15)

            cv.imwrite(str(os.path.join(flow_frames_path, 'frame-{}.png'.format(index))), img_flow_as_vectors)
            index += 1

        logger.info("Done estimating flow with deepflow.")


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

                # gpu_magnitude, gpu_angle = cv.cuda.cartToPolar(gpu_flow_x, gpu_flow_y, angleInDegrees=False)
                # gpu_v = cv.cuda.normalize(gpu_magnitude, 0.0, 255.0, cv.NORM_MINMAX, -1)
                # res = gpu_v.download()

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

   
    # https: // learnopencv.com / getting - started - opencv - cuda - module /
    # # convert from cartesian to polar coordinates to get magnitude and angle
    # magnitude, angle = cv2.cartToPolar(
    #     flow[..., 0], flow[..., 1], angleInDegrees=True,
    # )
    #
    # # set hue according to the angle of optical flow
    # hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))
    #
    # # set value according to the normalized magnitude of optical flow
    # hsv[..., 2] = cv2.normalize(
    #     magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1,
    # )
    #
    # # multiply each pixel value to 255
    # hsv_8u = np.uint8(hsv * 255.0)
    #
    # # convert hsv to bgr
    # bgr = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    #a_file_path = os.path.join("videos/", "flowformer.mp4")
    a_file_path = os.path.join("videos/", "optic_flow_snippet.mp4")
    #a_file_path = os.path.join("videos/", "demo_frames.mp4")
    source = flow_source(a_file_path)
    #source.extract_frames()

    #source.estimate_flow_with_raft()
    #source.write_video_from_raft()

    #source.estimate_flow_with_deepflow()

    source.estimate_flow_with_nvflow()
    source.write_video_from_nvflow()

    # source.estimate_flow_with_brox()
    # source.write_video_from_brox()

