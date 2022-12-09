'''Computes optic flow from one or more videos using openCV (deepflow or Farneback)
One video is single-threaded, multiple videos are multi-threaded (all CPU cores used).

Requires opencv installation with unofficial libraries: pip3 install opencv_contrib_python
'''
import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing
from itertools import product
from scipy.io import savemat


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
    tipSize = cv.norm(ptsDiff)*tipLength

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


def dispOpticalFlow(Image, Flow, Divisor, name):
    '''Display image with a visualisation of a flow over the top.
    A divisor controls the density of the quiver plot.'''

    PictureShape = np.shape(Image)
    # determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    # create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
        for j in range(1, Jmax):
            X1 = i*Divisor
            Y1 = j*Divisor
            X2 = int(X1 + Flow[X1, Y1, 1])
            Y2 = int(Y1 + Flow[X1, Y1, 0])
            X2 = np.clip(X2, 0, PictureShape[0])
            Y2 = np.clip(Y2, 0, PictureShape[1])
            # add all the lines to the mask
            mask = arrowedLine(mask, (X1, Y1), (X2, Y2), (255, 255, 0), 1)

    # superpose lines onto image
    img = cv.add(Image, mask)
    plt.imshow(img)
    plt.title(name)
    plt.pause(0.05)


def saveFlow(flow, imageNum, saveFolder):
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    imageNameDxDy = os.path.join(saveFolder, f'Frame{imageNum:04}.mat')

    # We need to negate the y flow component to go from image to cartesian (origin at top)
    flow[:, :, 1] = -flow[:, :, 1]

    # We need to reverse the flow vector row positions (origin top->bottom)
    flow = np.flipud(flow)

    # We assume (x, y) format for model in 1st cartesian quadrant
    # Currently, we have (row, col) format.
    # Since cols=x and rows=y, we need to transpose the rows/cols
    flow = np.transpose(flow, axes=(1, 0, 2))

    flowDict = {"dx": flow[..., 0], "dy": flow[..., 1], "dims": flow.shape[:2],
                "format": "videoFrame", "coordSystem": "cartesian1stQuadrant", "order": 'matlab',
                "fileNumDigits": 4}
    savemat(imageNameDxDy, flowDict, format='5', do_compression=True)


def processFlow(export_dir, video_filename, maxFrameNum=9999, res=[512, 512], displayFlow=False, method='deepflow'):
    res = np.array(res)

    video_name = os.path.basename(video_filename)
    video_name = os.path.splitext(video_name)[0]
    save_dir = os.path.join(export_dir, video_name, 'OpticFlow')

    print(f'Processing {video_name} for {maxFrameNum} frames maximum...')
    print(f'  Exporting to {save_dir}...')

    cap = cv.VideoCapture(video_filename)

    # Get frame 1
    ret, frame1 = cap.read()
    # Convert Frame 1 to grayscale
    prvsFrame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    # Resize Frame 1 as needed
    print (res, prvsFrame.shape)
    if np.any(res != prvsFrame.shape):
        prvsFrame = cv.resize(prvsFrame, dsize=tuple(res.tolist()))

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    prev = np.zeros(res)
    imNum = 1

    df = cv.optflow.createOptFlow_DeepFlow()

    # Read the next frame
    ret, frame2 = cap.read()

    # Get out if we don't have any more frames to process
    while frame2 is not None and imNum <= maxFrameNum:
        nextFrame = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        # Resize Next Frame as needed
        if np.any(res != nextFrame.shape):
            nextFrame = cv.resize(nextFrame, dsize=tuple(res.tolist()))

        # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        nextFrame = clahe.apply(nextFrame)

        time1 = time.time()
        if method == 'deepflow':
            flow = df.calc(prvsFrame, nextFrame, flow=None)
        else:
            flow = cv.calcOpticalFlowFarneback(prvsFrame, nextFrame, flow=None,
                                               pyr_scale=0.75, levels=12, winsize=15,
                                               iterations=4, poly_n=9, poly_sigma=1.75, flags=0)
        saveFlow(flow, imNum, save_dir)
        imNum = imNum + 1

        if imNum == 2:
            print(f'  One frame of OF detection took {time.time() - time1:.2f} sec')

        if displayFlow:
            dispOpticalFlow(prvsFrame, flow, 8, 'Image and Flow')

        # Get the next frame
        prvsFrame = nextFrame
        ret, frame2 = cap.read()

    print(f'Done with {video_name} after processing {imNum-1} frames!')
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    print('Num args:', len(sys.argv))
    if len(sys.argv) < 3:
        print('Required Usage: python3 openCVOF.py <exportDir> <fileContainingVideoNames> optionalArgs')
        print('Required Usage: python3 openCVOF.py <exportDir> <fullPathVideoFilename> optionalArgs')
        print('optionalArgs: <maxNumFrames> <ResRowsxResCols> <visualize01>')
        print('Example <ResRowsxResCols>: 512x512')
        exit()

    # Where we save the Optic Flow files
    export_dir = sys.argv[1]
    # Where we get the video(s)
    filename = sys.argv[2]

    # How many frames we process
    maxFrameNum = 9999
    if len(sys.argv) > 3:
        maxFrameNum = int(sys.argv[3])

    # At what resolution should we detect the optic flow?
    res = [512, 512]
    if len(sys.argv) > 4:
        res = sys.argv[4]
        res = res.split('x')
        res = [int(dim) for dim in res]
        print (sys.argv)

    # Whether we show a visualization of the optic flow as it is being detected
    visualizeOF = False
    if len(sys.argv) > 5 and not filename.endswith('.txt'):
        visualizeOF = int(sys.argv[5])
        if visualizeOF == 1:
            visualizeOF = True

    # Is this a file containing a list of video files?
    if filename.endswith('txt'):
        with open(filename, 'r') as fp:
            files = fp.readline().split(',')
            files = [file.strip() for file in files]
    else:
        files = [filename]
    # If we only are processing one video, just call the function like usual
    # If we have more than one, use multicore.
    if len(files) == 1:
        processFlow(export_dir, files[0], maxFrameNum=maxFrameNum, displayFlow=visualizeOF, res=res)
    else:
        # Assuming hyperthreading so half as many cores useable
        n_cores = multiprocessing.cpu_count() // 2

        with multiprocessing.Pool(n_cores) as p:
            p.starmap(processFlow, product([export_dir], files, [maxFrameNum], [res], [visualizeOF]))
