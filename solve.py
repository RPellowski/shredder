import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import color
from skimage import draw
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
    if False:
        fig, ax = plt.subplots(1,figsize=(14,7))
        ax.imshow(I)
        plt.show()

# Background modification specific to puzzle 1
#        Y0    Y1    X0    X1    WY  WX
BG = [(  40,   40,  200, 1300, 420,   0),
      ( 835,  900, 3388, 3423,   5,   5),
      ( 829,  824, 3391, 3427,   3,   3),
      (1189, 1275,  180,  151,   3,   3),
      (2186, 2281,  948,  940,   3,   3),
      (3177, 3177, 2430, 2518,   3,   3),
      (3460, 3520,  378,  368,   3,   3),
      (3464, 3424, 1818, 1919,   3,   3),
      (3447, 3455, 2464, 2608,   3,   3),
      ]

def clean_source_image():
    for (r0, r1, c0, c1, wy, wx) in BG:
        while wx > 0 or wy > 0:
            rr, cc, _ = draw.line_aa(r0, c0, r1, c1)
            I[rr, cc] = [255, 0, 150]
            if wx > 0:
                wx -= 1
                c0 += 1
                c1 += 1
            if wy > 0:
                wy -= 1
                r0 += 1
                r1 += 1

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
            M = copy.copy(masks[mask][y1:y2+1, x1:x2+1])
            M[LP != label] = 0
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

def get_orientation_lines():
    '''
    Create lines array
    '''
    lines = probabilistic_hough_line(M,
                                     threshold=h_threshold,
                                     line_length=h_line_length,
                                     line_gap=h_line_gap)
    angles = []
    for line in lines:
        p0, p1 = line
        if cur_mask == 'redlines':
            num = 0 - (p1[0] - p0[0])
            den = p1[1] - p0[1]
        else:
            num = p1[1] - p0[1]
            den = p1[0] - p0[0]
        if den == 0:
            theta = 90.0
        else:
            theta = atan(1.0 * num / den) * 180.0 / pi
        angles.append(theta)
    if len(lines) == 0:
        angles = [0]
    return (lines, angles)

def update(val):
    global litxt, aatxt
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
        litxt.remove()
        aatxt.remove()
        (lines, angles) = get_orientation_lines()
        for line in lines:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        aa = np.median(angles)
        litxt = ax[3].text(0, Y1 - 3*YD, "Lines:" + str(len(lines)))
        aatxt = ax[3].text(0, Y1 - 4*YD, "Med Angle:" + str(int(aa)))

def init_orientation_threshold():
    global h_threshold, h_line_length, h_line_gap
    h_threshold, h_line_length, h_line_gap = (10, 40, 35)

def setup_orientation_plot():
    global fig, ax
    global thamp, llamp, lgamp
    global thbar, llbar, lgbar, litxt, aatxt
    global X1, Y1, YD, W1, H1
    fig, ax = plt.subplots(1,4,figsize=(14,7))
    ax[0].imshow(P)
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

    thamp = Slider(thbar, 'Threshold',   1., 100., valinit=h_threshold,
            valfmt="%.0f")
    llamp = Slider(llbar, 'Line Length', 1., 300., valinit=h_line_length,
            valfmt="%.0f")
    lgamp = Slider(lgbar, 'Line Gap',    1., 300., valinit=h_line_gap,
            valfmt="%.0f")
    update(None)
    thamp.on_changed(update)
    llamp.on_changed(update)
    lgamp.on_changed(update)
    plt.show()

def get_orientation_angles(mask="bluelines", interactive=False):
    '''
    For each piece, examine the mask and if there are enough pixels, evaluate
    lines using the hough transform.  If there are enough lines, determine its
    orientation and mark the corresponding metadata as True.
    '''
    global min_pixels_for_orientation, min_lines_for_orientation
    global M, P
    global cur_mask, cur_label
    d = dict(labelinfo)
    min_pixels_for_orientation = 50
    min_lines_for_orientation = 2
    cur_mask = mask
    if mask == 'redlines':
        top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_rline_pix, reverse=True)
    elif mask == 'bluelines':
        top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_bline_pix, reverse=True)
    else:
        raise
    for i in range(len(top)):
        cur_label = top[i].label
        if mask == 'redlines':
            count = Pieces[(cur_label, 0)].src_n_rline_pix
        else:
            count = Pieces[(cur_label, 0)].src_n_bline_pix
        (y1, x1, y2, x2) = d[cur_label]
        P = copy.copy(I[y1:y2+1, x1:x2+1])
        LP = label_image[y1:y2+1, x1:x2+1]
        P[LP != cur_label] = 0
        M = copy.copy(masks[cur_mask][y1:y2+1, x1:x2+1])
        M[LP != cur_label] = 0
        if count <= min_pixels_for_orientation:
            # No more pieces that qualify
            break
        init_orientation_threshold()
        if interactive:
            setup_orientation_plot()
        (lines, angles) = get_orientation_lines()
        if len(lines) >= min_lines_for_orientation:
            aa = np.median(angles)
            Pieces[(cur_label, 0)].dst_b_angle = True
            Pieces[(cur_label, 0)].dst_angle = aa
            #print "label:", cur_label, "angle:", aa, lines, angles
    #ok = sum(Piece.dst_b_angle for Piece in Pieces.values())
    #not_ok = sum(not(Piece.dst_b_angle) for Piece in Pieces.values())
    #print len(Pieces), ok, not_ok
    if False:
        # Move the next lina after the aa assignment above
        # label_image[label_image == cur_label] = 0
        fig, ax = plt.subplots(1,figsize=(14,7))
        IX = copy.copy(I)
        IX[label_image == 0] = 0
        ax.imshow(IX)
        plt.show()

if __name__ == '__main__':
    #io.find_available_plugins() #
    global logger
    global masks
    logger = logmetrics.initLogger()
    t0 = logmetrics.unix_time()
    try:
        open_source_image()
        clean_source_image()
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
        get_orientation_angles("redlines")
        get_orientation_angles("bluelines")
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt")
    except:
        raise
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))
else:
    open_source_image()
