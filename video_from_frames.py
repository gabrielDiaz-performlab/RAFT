import os
import sys

import av
import glob


from natsort import natsorted, ns

sys.path.append('core')

from PIL import Image

import logging
logger = logging.getLogger(__name__)
# These lines allow me to see logging.info messages in my jupyter cell output
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)

frames_dir = "D:\\Github\\RAFT\\videos\\sintel_training_final\\mountain_1"
video_out_path = "D:\\Github\\RAFT\\videos\\"
video_out_filename = "mountain_1.mp4"

fps = 24

video_out_path = os.path.join( video_out_path, video_out_filename )
logger.info("Writing movie...")

if os.path.isdir(video_out_path) is False:
    os.makedirs(video_out_path)

images = glob.glob(os.path.join(frames_dir, '*.png')) + \
         glob.glob(os.path.join(frames_dir, '*.jpg'))

images = natsorted(images, key=lambda y: y.lower())  # sort alphanumeric in ascending order

total_frames = len(images)
container = av.open(os.path.join(video_out_path, video_out_filename), mode="w")

img = Image.open(images[0])
stream = container.add_stream("libx264", rate=fps)
stream.options["crf"] = "15"

# stream.pix_fmt = "yuv444p"
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