import os
import sys
import numpy as np
import av

sys.path.append('core')
from PIL import Image

os.add_dll_directory("D://opencvgpu//opencv_build_310//bin")
os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.8//bin")

import cv2

from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)
# These lines allow me to see logging.info messages in my jupyter cell output
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)

import pickle
from tqdm import tqdm


class flow_source():

    def __init__(self, file_path, out_parent_dir="frames_out"):

        self.file_path = file_path
        self.source_file_name = os.path.split(file_path)[-1].split('.')[0]
        self.source_file_suffix = os.path.split(file_path)[-1].split('.')[1]

        self.out_parent_dir = out_parent_dir
        self.raw_frames_path = os.path.join(out_parent_dir, self.source_file_name, 'raw')
        self.video_out_path = os.path.join(out_parent_dir, self.source_file_name, 'videos')

    def view_mag_histogram(self, video_out_name):

        pickle_loc = os.path.join(self.video_out_path, video_out_name.split('.')[0] + '_mag.pickle')
        pickle_file = open(pickle_loc, 'rb')
        mag_dict = pickle.load(pickle_file)

        mag_values = mag_dict['values']
        bins = mag_dict['bins']

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        # ax.bar(bins[:-1], mag_values, width = .75 * (bins[1]-bins[0]))
        ax.bar(bins[:-1], np.cumsum(mag_values) / sum(mag_values) , width = .75 * (bins[1]-bins[0]))

        # tidy up the figure
        ax.grid(True)

        ax.set_title('flow magnitude')
        ax.set_xlabel('value')
        ax.set_ylabel('likelihood')

        plt.savefig( os.path.join(self.video_out_path, video_out_name.split('.')[0] + '_mag.jpg') )

    def calculate_flow(self, video_out_name, algorithm = "deepflow", visualize_as="hsv_stacked", hist_params = (100, 0,40), lower_mag_threshold = False, upper_mag_threshold = False):

        container_in = av.open(self.file_path)
        average_fps = container_in.streams.video[0].average_rate
        num_frames = container_in.streams.video[0].frames

        ##############################
        # prepare video out
        if os.path.isdir(self.video_out_path) is False:
            os.makedirs(self.video_out_path)

        container_out = av.open(os.path.join(self.video_out_path, video_out_name), mode="w")
        stream = container_out.add_stream("libx264", rate=average_fps)
        stream.options["crf"] = "15"

        # this_frame = container_in.decode(video=0)

        ##############################
        use_cuda = False

        if algorithm == "deepflow":
            flow_algo = cv2.optflow.createOptFlow_DeepFlow()

        elif algorithm == "tvl1":

            flow_algo = cv2.cuda_OpticalFlowDual_TVL1.create()
            flow_algo.setLambda(0.1) # default 0.15. smaller = smoother output.

            # Epsilon: Stopping criterion threshold used in the numerical scheme, which is a trade-off between precision and
            # running time. A small value will yield more accurate solutions at the expense of a slower convergence.
            # Default is 0.01
            # flow_algo.setEpsilon(0.005)

            use_cuda = True
            image1_gpu = cv2.cuda_GpuMat()
            image2_gpu = cv2.cuda_GpuMat()

        else:
            logger.error('Optical flow algorithm not yet implemented.')

        idx = 0
        for frame in tqdm(container_in.decode(video=0), desc='Calculating flow', unit= 'frames',total = num_frames):

            if(idx == 0):

                stream.width = frame.width
                stream.height = frame.height

                # frame.to_image().save(os.path.join(self.raw_frames_path, '{:06d}.png'.format(frame.index)))
                prev_frame = frame.to_ndarray(format='bgr24')
                idx += 1
                continue
            else:
                frame = frame.to_ndarray(format='bgr24')

            # Calculate flow.  If possible, use cuda.
            if use_cuda:

                image1_gpu.upload(frame)
                image2_gpu.upload(prev_frame)

                image1_gray = cv2.cuda.cvtColor(image1_gpu, cv2.COLOR_BGR2GRAY)
                image2_gray = cv2.cuda.cvtColor(image2_gpu, cv2.COLOR_BGR2GRAY)

                flow = flow_algo.calc(image1_gray, image2_gray, flow=None)

                image1_gray = image1_gray.download()
                flow = flow.download()

            else:
                image1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image2_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                flow = flow_algo.calc(image1_gray, image2_gray, flow=None)

            frameout = []

            # Convert flow to mag / angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

            if visualize_as == "vectors":

                # if algorithm == "tvl1":
                #     flip_direction = True
                # else:
                #     flip_direction = False

                #vector_flow = self.visualize_flow_as_vectors_old(frame, flow, 15)
                vector_flow = self.visualize_flow_as_vectors(frame, magnitude, angle, 10) #, magnitude_scalar=-1)

                frameout = av.VideoFrame.from_ndarray(vector_flow, format='bgr24')

            elif visualize_as == "hsv_overlay" or visualize_as == "hsv_stacked":

                hsv_flow = self.visualize_flow_as_hsv(magnitude, angle, lower_mag_threshold, upper_mag_threshold)

                if visualize_as == "hsv_overlay":
                    combined_image = cv2.addWeighted(cv2.cvtColor(image1_gray, cv2.COLOR_GRAY2BGR), 0.1, hsv_flow, 0.9, 0)
                elif visualize_as == "hsv_stacked":
                    combined_image = np.concatenate((frame, hsv_flow), axis=0)
                else:
                    logger.error('Visualization method not recognized.')

                frameout = av.VideoFrame.from_ndarray(combined_image, format='bgr24')


            else:
                logger.exception('visualize_as string not supported')

            # Add packet to video
            for packet in stream.encode(frameout):
                container_out.mux(packet)

            # Store the histogram of avg magnitudes
            if idx == 1:
                mag_hist = np.histogram(magnitude, hist_params[0], (hist_params[1], hist_params[2]))
                # Store the first flow histogram
                cumulative_mag_hist = mag_hist[0]
            else:
                # Calc cumulative avg flow magnitude by adding the first flow histogram in a weighted manner
                cumulative_mag_hist = np.divide(
                    np.sum([np.multiply((idx - 1), cumulative_mag_hist), mag_hist[0]], axis=0), idx - 1)

            prev_frame = frame
            idx += 1

        # Flush stream
        for packet in stream.encode():
            container_out.mux(packet)

        # Close the file
        container_out.close()

        dbfile = open(os.path.join(self.video_out_path, video_out_name.split('.')[0] + '_mag.pickle'), 'wb')
        pickle.dump( {"values": cumulative_mag_hist, "bins": mag_hist[1]}, dbfile)
        dbfile.close()

    def visualize_flow_as_hsv(self, magnitude, angle, lower_mag_threshold = False, upper_mag_threshold = False):

        # convert from cartesian to polar coordinates to get magnitude and angle
        # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # create hsv output for optical flow
        # (480, 640, 3)
        hsv = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.float32)

        # set hue according to the angle of optical flow
        hsv[..., 0] = angle * ((1 / 360.0) * (180 / 255.0))

        # set saturation to 1
        hsv[..., 1] = 1.0

        if lower_mag_threshold:
            magnitude[magnitude<lower_mag_threshold] = 0

        if upper_mag_threshold:
            magnitude[magnitude>upper_mag_threshold] = 0
            hsv[..., 2] = np.clip(magnitude / upper_mag_threshold, 0, 1)
        else:
            hsv[..., 2] = cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX, -1)

        # multiply each pixel value to 255
        hsv_8u = np.uint8(hsv * 255.0)

        # convert hsv to rgb
        # rgb = cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

        # combined = cv2.addWeighted(frame, 0.05, cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR), 0.95, 0)
        return cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    def visualize_flow_as_vectors(self, frame, magnitude, angle, divisor, lower_mag_threshold = False, upper_mag_threshold = False, magnitude_scalar = False):

        '''Display image with a visualisation of a flow over the top.
        A divisor controls the density of the quiver plot.'''

        # create a blank mask, on which lines will be drawn.
        mask = np.zeros([np.shape(magnitude)[0], np.shape(magnitude)[1], 3], np.uint8)

        # (2, 480, 640)
        vector_x, vector_y = cv2.polarToCart(magnitude, angle, angleInDegrees=True)

        for r in range(1, int(np.shape(magnitude)[0] / divisor)):
            for c in range(1, int(np.shape(magnitude)[1] / divisor)):

                origin_x = r * divisor
                origin_y = c * divisor

                endpoint_x = int(origin_x + vector_x[origin_x, origin_y])
                endpoint_y = int(origin_y + vector_y[origin_x, origin_y])

                mask = cv2.arrowedLine(mask, (origin_y, origin_x), (endpoint_y, endpoint_x),  (0, 255, 255), 1)

        return cv2.add(frame, mask)



if __name__ == "__main__":

    # a_file_path = os.path.join("videos/", "640_480_60Hz_small.mp4")
    # source = flow_source(a_file_path)
    # source.calculate_flow('640_480_60Hz_small_deepflow_vectors.mp4', visualize_as="vectors",lower_mag_threshold = 2, upper_mag_threshold = 40)
    # source.view_mag_histogram('640_480_60Hz_small_deepflow_vectors.mp4')

    # a_file_path = os.path.join("videos/", "1280_960_30Hz.mp4")
    # source = flow_source(a_file_path)
    # source.(('1280_960_30Hz_deepflow.mp4')

    a_file_path = os.path.join("videos/", "640_480_60Hz_small.mp4")
    source = flow_source(a_file_path)
    source.calculate_flow('640_480_60Hz_small_tvl1_vectors.mp4',algorithm='tvl1',visualize_as="vectors", lower_mag_threshold = 0.5, upper_mag_threshold = 30)
    source.view_mag_histogram('640_480_60Hz_small_tvl1_vectors.mp4')

    # a_file_path = os.path.join("videos/", "640_480_60Hz.mp4")
    # source = flow_source(a_file_path)
    # source.calculate_flow('640_480_60Hz_tvl1.mp4',algorithm='tvl1', lower_mag_threshold = 0.5, upper_mag_threshold = 20)
    # source.view_mag_histogram('640_480_60Hz_tvl1.mp4')

    # a_file_path = os.path.join("videos/", "1280_960_30Hz.mp4")
    # source = flow_source(a_file_path)
    # source.calculate_flow('1280_960_30Hz_tvl1.mp4',algorithm='tvl1', lower_mag_threshold = 0.5, upper_mag_threshold = 15)
    # source.view_mag_histogram('1280_960_30Hz_tvl1.mp4')


