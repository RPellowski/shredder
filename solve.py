import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import color
from skimage import img_as_float
import logging, logmetrics
from piece import *
import copy
import os
import sys
from math import atan2, pi, atan

DPI = 400
def open_source_image():
    global I
    I=io.imread('puzzle1_400dpi.tif')

def calculate_source_stats():
    logger.info(
        "Image shape: {} Image size: {} Image datatype: {} Area: {} sq in".format(
        I.shape, I.size, I.dtype, float(I.shape[0] * I.shape[1])/DPI/DPI)
        )

def create_mask(mask):
    fname = "cache/mask_" + mask + '.npy'
    if os.path.isfile(fname):
        logger.info("Reading mask {}".format(fname))
        m = np.load(fname)
    else:
        logger.info("Creating mask {}".format(fname))
        m = np.zeros(I.shape[:-1], dtype=bool)
        HL = HSL_PARAMS[mask][0] - HSL_PARAMS[mask][3]
        HU = HSL_PARAMS[mask][0] + HSL_PARAMS[mask][3]
        if HL < 0.0:
            HL2 = HL + 1.0
        else:
            HL2 = 1.0
        if HU > 1.0:
            HU2 = HU - 1.0
        else:
            HU2 = 0.0
        SL = HSL_PARAMS[mask][1] - HSL_PARAMS[mask][4]
        SU = HSL_PARAMS[mask][1] + HSL_PARAMS[mask][4]
        LL = HSL_PARAMS[mask][2] - HSL_PARAMS[mask][5]
        LU = HSL_PARAMS[mask][2] + HSL_PARAMS[mask][5]

        # Have to do it line by line due to memory limitations
        for y in range(I.shape[0]):
            iline = img_as_float(I[y:y+1,:,:])
            ihsv = color.rgb2hsv(iline)
            m[y] = \
                (((ihsv[:,:,0] > HL) & (ihsv[:,:,0] < HU)) |
                    (ihsv[:,:,0] > HL2) |
                    (ihsv[:,:,0] < HU2)) & \
                (ihsv[:,:,1] > SL) & \
                (ihsv[:,:,1] < SU) & \
                (ihsv[:,:,2] > LL) & \
                (ihsv[:,:,2] < LU)
            # Very specific to puzzle 1
            if mask=="pieces":
                if y < 500:
                    m[y,:1300] = False
        percentage = np.count_nonzero(m)*100.0/m.size
        logger.debug({mask:str(percentage)})
        np.save(fname, m)
    return m

def label_blobs():
    from skimage.morphology import closing, square
    from skimage.measure import label, regionprops

    I1 = copy.copy(I)
    I1[masks["pieces"]] = 255
    I1[~masks["pieces"]] = 0
    gray = I1[:,:,0]
    bw = closing(gray > 128, square(3))
    percentage = np.count_nonzero(bw)*100.0/bw.size
    logger.debug({"closing":str(percentage)})
    label_image = label(bw)
    properties = regionprops(label_image)
    pieces = 0
    labelinfo = []
    for proper in properties:
        if proper.area > 100:
            pieces += 1
            labelinfo.append((proper.label, proper.bbox))
    logger.info("Labeled pieces: {}".format(pieces))
    if False:
        fig, ax = plt.subplots(1,figsize=(14,7))
        ax.imshow(I)
        for _,bbox in labelinfo:
            (y1, x1, y2, x2) = bbox
            c = plt.Rectangle((int(x1),int(y1)), int(x2-x1), int(y2-y1),
                linewidth=1, edgecolor="white", facecolor='none')
            ax.add_patch(c)
        plt.show()
    return (label_image, labelinfo)

def generate_pieces_and_metadata(label_image, labelinfo):
    for linf in labelinfo:
        label, bbox = linf
        (y1, x1, y2, x2) = bbox
        P = copy.copy(I[y1:y2+1, x1:x2+1])
        LP = label_image[y1:y2+1, x1:x2+1]
        P[LP != label] = 0

        # Save piece data as an np array
        fname = "cache/src_" + str(label) + '.npy'
        if not os.path.isfile(fname):
            logger.info("Creating: {}".format(fname))
            np.save(fname, P)

        # Metadata starts here
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # P contains the pixels that are not background
        piece = Piece(label, x1, y1, w, h)

        # From these, select the ones we think are useful
        misc = np.ones_like(LP)
        misc[LP != label] = 0
        piece_pix = 0
        for mask in ["bluelines" ,"redlines", "blackink", "paper"]:
            M = masks[mask][y1:y2+1, x1:x2+1]
            misc[M>0] = 0
            pix_count = np.count_nonzero(M)
            piece_pix += pix_count
            if mask == "bluelines":
                piece.src_n_bline_pix = pix_count
            elif mask == "redlines":
                piece.src_n_rline_pix = pix_count
            elif mask == "blackink":
                piece.src_n_bink_pix = pix_count
            elif mask == "paper":
                piece.src_n_paper_pix = pix_count
        piece.src_n_misc_pix = np.count_nonzero(misc)
        piece_pix += piece.src_n_misc_pix
        piece.src_n_bg_pix = w * h - piece_pix

        # Save piece metadata as a json array
        mname = "cache/meta_" + str(label) + '.json'
        if not os.path.isfile(mname):
            logger.info("Creating: {}".format(mname))
            with open(mname, "w") as metafile:
                metafile.write(str(piece))
        Pieces[(label, piece.candidate)] = piece

def show_top_pieces(red=False, blue=False, black=False):
    # consume label:bbox as dict
    d = dict(labelinfo)

    # Now for red line test ----------------------
    # get top red line pieces
    top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_rline_pix, reverse=True)
    # display top red line pieces
    if red:
        SHOW=20
        fig, ax = plt.subplots(2,SHOW,figsize=(14,7))
        for i in range(SHOW):
            label = top[i].label
            print label,top[i].src_n_rline_pix, d[label]
            (y1, x1, y2, x2) = d[label]
            P = I[y1:y2+1, x1:x2+1]
            LP = label_image[y1:y2+1, x1:x2+1]
            P[LP != label] = 0
            M = masks["redlines"][y1:y2+1, x1:x2+1]
            ax[0,i].imshow(P)
            ax[1,i].imshow(M)
        plt.show()

    # Now for blue line test ----------------------
    # get top blue line pieces
    top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_bline_pix, reverse=True)
    # display top blue line pieces
    if blue:
        SHOW=20
        fig, ax = plt.subplots(2,SHOW,figsize=(14,7))
        for i in range(SHOW):
            label = top[i].label
            print label,top[i].src_n_bline_pix, d[label]
            (y1, x1, y2, x2) = d[label]
            P = I[y1:y2+1, x1:x2+1]
            LP = label_image[y1:y2+1, x1:x2+1]
            P[LP != label] = 0
            M = masks["bluelines"][y1:y2+1, x1:x2+1]
            ax[0,i].imshow(P)
            ax[1,i].imshow(M)
        plt.show()

    # Now for black ink test ----------------------
    # get top black ink pieces
    top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_bink_pix, reverse=True)
    # display top black ink pieces
    if black:
        SHOW=20
        fig, ax = plt.subplots(2,SHOW,figsize=(14,7))
        for i in range(SHOW):
            label = top[i].label
            print label,top[i].src_n_bink_pix, d[label]
            (y1, x1, y2, x2) = d[label]
            P = I[y1:y2+1, x1:x2+1]
            LP = label_image[y1:y2+1, x1:x2+1]
            P[LP != label] = 0
            M = masks["blackink"][y1:y2+1, x1:x2+1]
            ax[0,i].imshow(P)
            ax[1,i].imshow(M)
        plt.show()

from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage.feature import canny
from matplotlib.widgets import Slider, Button, RadioButtons

def update(val):
    global thamp, llamp, lgamp
    global thbar, llbar, lgbar, litxt, aatxt
    global h_threshold, h_line_length, h_line_gap
    global IM
    if val is None:
        ax[1].imshow(M)
        IM = ax[2].imshow(M)
    if True:
        h_threshold = int(thamp.val)
        h_line_length = int(llamp.val)
        h_line_gap = int(lgamp.val)
        IM.set_data(M)
        while len(ax[2].lines) > 0:
            del ax[2].lines[0]
        lines = probabilistic_hough_line(M,
                                         threshold=h_threshold,
                                         line_length=h_line_length,
                                         line_gap=h_line_gap)
        aa = 0.0
        aaa = []
        for line in lines:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
            aa += atan2(p1[1] - p0[1], p1[0] - p0[0])
            #print atan(1.0*(p1[1] - p0[1])/( p1[0] - p0[0])) * 180.0 / pi
            if p1[0] == p0[0]:
                theta = 90.0
            else:
                theta = atan(1.0*(p1[1] - p0[1])/(p1[0] - p0[0])) * 180.0 / pi
            aaa.append(theta)
        if len(lines) > 0:
            aa = int(aa * 180.0 / pi / len(lines))
        else:
            aaa=[0]
        litxt.remove()
        aatxt.remove()
        #print np.median(aaa)
        litxt  = ax[3].text(0, Y1 - 3*YD, "Lines:" + str(len(lines)))
        aatxt  = ax[3].text(0, Y1 - 4*YD, "Med Angle:" + str(int(np.median(aaa))))

def get_orientation_angles():
    SHOW=30
    global fig, ax
    global M
    global h_threshold, h_line_length, h_line_gap
    global thamp, llamp, lgamp
    global thbar, llbar, lgbar, litxt, aatxt
    global X1, Y1, YD, W1, H1
    d = dict(labelinfo)
    top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_rline_pix, reverse=True)
    mask = masks["redlines"]
    #top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_bline_pix, reverse=True)
    #mask = masks["bluelines"]
    for i in range(SHOW):
        label = top[i].label
        (y1, x1, y2, x2) = d[label]
        M = copy.copy(mask[y1:y2+1, x1:x2+1])
        h_threshold, h_line_length, h_line_gap = (10,40,35)
        fig, ax = plt.subplots(1,4,figsize=(14,7))
        ax[0].imshow(copy.copy(I[y1:y2+1, x1:x2+1]))
        ax[3].axis('off')
        X1 = 0.8
        Y1 = 0.8
        YD = 0.07
        W1 = 0.15
        H1 = 0.03
        thbar  = plt.axes([X1, Y1 - 0*YD, W1, H1])
        llbar  = plt.axes([X1, Y1 - 1*YD, W1, H1])
        lgbar  = plt.axes([X1, Y1 - 2*YD, W1, H1])
        litxt  = ax[3].text(0, Y1 - 3*YD, "Lines")
        aatxt  = ax[3].text(0, Y1 - 4*YD, "Ave Angle")

        thamp = Slider(thbar, 'Threshold',   1., 100.0, valinit=h_threshold,
                valfmt="%.0f")
        llamp = Slider(llbar, 'Line Length', 1., 300.0, valinit=h_line_length,
                valfmt="%.0f")
        lgamp = Slider(lgbar, 'Line Gap',    1., 300.0, valinit=h_line_gap,
                valfmt="%.0f")

        update(None)
        thamp.on_changed(update)
        llamp.on_changed(update)
        lgamp.on_changed(update)

        plt.show()

if __name__ == '__main__':
    #io.find_available_plugins() #
    global logger
    global masks
    logger = logmetrics.initLogger()
    t0 = logmetrics.unix_time()
    try:
        open_source_image()
        calculate_source_stats()
        masks = {}
        for mask in HSL_PARAMS.keys():
            masks[mask] = create_mask(mask)
        if False:
            SHOW = 1 + len(masks)
            fig, ax = plt.subplots(1,SHOW,figsize=(14,7))
            ax[0].imshow(I[:2000,:2000,:])
            ax_index = 1
            for mask in masks:
                ax[ax_index].imshow(masks[mask][:2000,:2000])
                ax_index += 1
            plt.show()

        (label_image, labelinfo) = label_blobs()
        generate_pieces_and_metadata(label_image, labelinfo)
        #show_top_pieces(red=True, blue=True, black=True)
        get_orientation_angles()
    except:
        logger.debug("Encountered error")
        raise
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))
else:
    theta = np.pi / 2 - np.arange(180) / 180.0 * np.pi
    #theta = np.arange(180)
    print len(theta)
    print theta *180/np.pi
