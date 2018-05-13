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
from collections import *

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

from skimage.morphology import closing, square
from skimage.measure import label, regionprops

def label_blobs():
    global mylabels
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
    labelinfo = {}
    for proper in properties:
        if proper.area > 100:
            pieces += 1
            labelinfo[proper.label] = proper.bbox
    logger.info("Labeled pieces: {}".format(pieces))
    mylabels = sorted(labelinfo.keys())
    if False:
        fig, ax = plt.subplots(1,figsize=(14,7))
        ax.imshow(I)
        for _,bbox in labelinfo.items():
            (y1, x1, y2, x2) = bbox
            c = plt.Rectangle((int(x1),int(y1)), int(x2-x1), int(y2-y1),
                linewidth=1, edgecolor="white", facecolor='none')
            ax.add_patch(c)
        plt.show()
    return (label_image, labelinfo)

def generate_pieces_and_metadata(label_image, labelinfo):
    for label, bbox in labelinfo.items():
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
    d = labelinfo

    # Now for red line test ----------------------
    # get top red line pieces
    top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_rline_pix, reverse=True)
    # display top red line pieces
    if red:
        SHOW=20
        fig, ax = plt.subplots(2,SHOW,figsize=(14,7))
        for i in range(SHOW):
            label = top[i].label
            #print label,top[i].src_n_rline_pix, d[label]
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
            #print label,top[i].src_n_bline_pix, d[label]
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
            #print label,top[i].src_n_bink_pix, d[label]
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
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

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
    (X1, Y1, YD, W1, H1) = (0.8, 0.8, 0.07, 0.15, 0.03)
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
    d = labelinfo
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
        if Pieces[(cur_label, 0)].dst_b_angle:
            continue
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
            Pieces[(cur_label, 0)].set_b_angle(True)
            Pieces[(cur_label, 0)].set_angle(aa)
            #print "label:", cur_label, "angle:", aa, lines, angles

def show_orientation_stats():
    stats = defaultdict(int)
    EPS = 1e-5
    for _, piece in Pieces.items():
        stats["b_angle"]    += int(piece.dst_b_angle)
        stats["angle"]      += int((abs(piece.dst_angle) - EPS) > 0)
        stats["b_polarity"] += int(piece.dst_b_polarity)
        stats["polarity"]   += int((abs(piece.dst_polarity) - EPS) > 0)
        stats["result"]     += int((abs(piece.dst_result) - EPS) > 0)
        stats["pieces"]     += 1
    for stat in "b_angle", "angle", "b_polarity", "polarity", "result":
        logger.info( \
            "{}: {}, {:.1f}%".format(stat, stats[stat], stats[stat] * 100.0 / stats["pieces"]))

from skimage import data, transform
def view_rotations(fake=False):
    if fake:
        global I
        I = data.checkerboard()
        p = Piece(42, 0, 0, I.shape[0], I.shape[1])
        p.set_b_angle(True)
        p.set_angle(10.0)
        Pieces[(42, 0)] = p
    for label, bbox in labelinfo.items():
        (y1, x1, y2, x2) = bbox
        P = copy.copy(I[y1:y2+1, x1:x2+1])
        LP = label_image[y1:y2+1, x1:x2+1]
        P[LP != label] = 0
        #for piece in Pieces.values():
        piece = Pieces[(label, 0)]
        #print piece.label, piece.dst_angle
        if not piece.dst_b_angle:
            fig, ax = plt.subplots(1,3,figsize=(14,7))
            IX = copy.copy(P)
            IX = transform.rotate(IX, piece.dst_angle, resize=True, mode='constant', cval=0)
            ax[0].imshow(P)
            ax[1].imshow(IX)
            ax[2].axis('off')
            plt.show()

def init_sr():
    global SRP
    global ignore_sr_update
    cur_label = mylabels[0]
    piece = Pieces[(cur_label, 0)]
    (X1, Y1, YD, W1, H1) = (0.1, 0.4, 0.07, 0.8, 0.03)
    # This dict holds parameters for setup of the visual elements
    SRP = {
        "iy"           : 2,
        "ix"           : 2,
        "fig"          : None,
        "ax"           : None,
        "axidx_map"    : 0,
        "axidx_piece"  : 1,
        "axidx_off"    : [2,3],

        "bar_label"    : [X1, Y1 - 0*YD, W1, H1],
        "info_label"   : ["Piece Label", mylabels[0], mylabels[-1], cur_label, "%.0f"],
        "slider_label" : None,

        "bar_angles"   : [0.2, Y1 - 2*YD, 0.07, 4*H1],
        "info_angles"  : [("True", "False", "All"), 2],
        "text_angles"  : [0.1, Y1 - 1*YD, "Show Angles:"],
        "radio_angles" : None,

        "bar_pols"     : [0.4, Y1 - 2*YD, 0.07, 4*H1],
        "info_pols"    : [("True", "False", "All"), 2],
        "text_pols"    : [0.3, Y1 - 1*YD, "Show Polarizations:"],
        "radio_pols"   : None,

        "bar_rot"      : [X1, Y1 - 3*YD, W1, H1],
        "info_rot"     : ["Rotation", -45., 315., piece.dst_result, "%.1f"],
        "slider_rot"   : None,

        "text_bangle"  : [0.1, Y1 - 4.5*YD, "Angle Bool: {!s}"],
        "wid_bangle"   : None,

        "text_angle"   : [0.1, Y1 - 5*YD, "Angle: {:.0f}"],
        "wid_angle"    : None,

        "text_bpol"    : [0.3, Y1 - 4.5*YD, "Pol Bool: {!s}"],
        "wid_bpol"     : None,

        "text_pol"     : [0.3, Y1 - 5*YD, "Pol: {:.0f}"],
        "wid_pol"      : None,

        "text_result"  : [0.5, Y1 - 5*YD, "Resultant Angle: {:.0f}"],
        "wid_result"   : None,

        "image_map"    : None,
        "aximg_map"    : None,
        #"image_piece"  : None,
        #"aximg_piece"  : None,

        "bar_prev"     : [0.6, Y1 - 1.5*YD, 0.05, 2*H1],
        "info_prev"    : ["Prev"],
        "btn_prev"     : None,

        "bar_next"     : [0.7, Y1 - 1.5*YD, 0.05, 2*H1],
        "info_next"    : ["Next"],
        "btn_next"     : None,

        "bar_save"     : [0.8, Y1 - 1.5*YD, 0.05, 2*H1],
        "info_save"    : ["Save"],
        "btn_save"     : None,

        "bar_b_bangle"   : [0.1, Y1 - 4*YD, 0.08, 2*H1],
        "info_b_bangle"  : ["Set b_angle"],
        "check_b_bangle" : None,

        "bar_b_bpol"   : [0.3, Y1 - 4*YD, 0.08, 2*H1],
        "info_b_bpol"  : ["Set b_pol"],
        "check_b_bpol" : None,

        "bar_invert"   : [0.4, Y1 - 4*YD, 0.08, 2*H1],
        "info_invert"  : ["Set invert"],
        "check_invert" : None,
        "cur_label"    : cur_label,
        }

    SRP["fig"], ax = plt.subplots(SRP["iy"],SRP["ix"],figsize=(14,7))
    SRP["ax"] = np.atleast_1d(ax.ravel())

    bar = plt.axes(SRP["bar_b_bangle"])
    SRP["check_b_bangle"] = CheckButtons(bar, SRP["info_b_bangle"], [0])

    bar = plt.axes(SRP["bar_b_bpol"])
    SRP["check_b_bpol"] = CheckButtons(bar, SRP["info_b_bpol"], [0])
    bar = plt.axes(SRP["bar_invert"])
    SRP["check_invert"] = CheckButtons(bar, SRP["info_invert"], [0])

    (y1, x1, y2, x2) = labelinfo[cur_label]
    # Show the image, I
    SRP["aximg_map"] = SRP["ax"][SRP["axidx_map"]].imshow(I)
    c = plt.Rectangle((int(x1),int(y1)), int(x2-x1), int(y2-y1),
        linewidth=1, edgecolor="white", facecolor='none')
    SRP["ax"][SRP["axidx_map"]].add_patch(c)

    # Create P, the piece
    P = copy.copy(I[y1:y2+1, x1:x2+1])
    LP = label_image[y1:y2+1, x1:x2+1]
    P[LP != SRP["cur_label"]] = 0
    # Rotate if indicated
    P = transform.rotate(P, piece.dst_result, resize=True, mode='constant', cval=0)
    SRP["aximg_piece"] = SRP["ax"][SRP["axidx_piece"]].imshow(P)

    for idx in SRP["axidx_off"]:
        SRP["ax"][idx].axis('off')

    bar = plt.axes(SRP["bar_label"])
    SRP["slider_label"] = Slider(bar, SRP["info_label"][0],
        SRP["info_label"][1], SRP["info_label"][2],
        valinit=SRP["info_label"][3],
        valfmt=SRP["info_label"][4])

    plt.text(SRP["text_angles"][0], SRP["text_angles"][1], SRP["text_angles"][2],transform=SRP["fig"].transFigure)
    bar = plt.axes(SRP["bar_angles"])
    SRP["radio_angles"] = RadioButtons(bar, SRP["info_angles"][0], active=SRP["info_angles"][1])

    plt.text(SRP["text_pols"][0], SRP["text_pols"][1], SRP["text_pols"][2],transform=SRP["fig"].transFigure)
    bar = plt.axes(SRP["bar_pols"])
    SRP["radio_pols"] = RadioButtons(bar, SRP["info_pols"][0], active=SRP["info_pols"][1])

    bar = plt.axes(SRP["bar_rot"])
    SRP["slider_rot"] = Slider(bar, SRP["info_rot"][0],
        SRP["info_rot"][1], SRP["info_rot"][2],
        valinit=SRP["info_rot"][3],
        valfmt=SRP["info_rot"][4])

    bar = plt.axes(SRP["bar_prev"])
    SRP["button_prev"] = Button(bar, SRP["info_prev"][0])

    bar = plt.axes(SRP["bar_next"])
    SRP["button_next"] = Button(bar, SRP["info_next"][0])

    bar = plt.axes(SRP["bar_save"])
    SRP["button_save"] = Button(bar, SRP["info_save"][0])

    # Specific piece calculations
    s = SRP["text_bangle"][2].format(piece.dst_b_angle)
    SRP["wid_bangle"] = plt.text(SRP["text_bangle"][0], SRP["text_bangle"][1], s, transform=SRP["fig"].transFigure)
    s = SRP["text_angle"][2].format(piece.dst_angle)
    SRP["wid_angle"] = plt.text(SRP["text_angle"][0], SRP["text_angle"][1], s, transform=SRP["fig"].transFigure)

    s = SRP["text_bpol"][2].format(piece.dst_b_polarity)
    SRP["wid_bpol"] = plt.text(SRP["text_bpol"][0], SRP["text_bpol"][1], s, transform=SRP["fig"].transFigure)
    s = SRP["text_pol"][2].format(piece.dst_polarity)
    SRP["wid_pol"] = plt.text(SRP["text_pol"][0], SRP["text_pol"][1], s, transform=SRP["fig"].transFigure)

    s = SRP["text_result"][2].format(piece.dst_result)
    SRP["wid_result"] = plt.text(SRP["text_result"][0], SRP["text_result"][1], s, transform=SRP["fig"].transFigure)

    ignore_sr_update = False

def update_sr_images():
    cur_label = SRP["cur_label"]
    if True:
        piece = Pieces[(cur_label, 0)]
        # do the full update
        (y1, x1, y2, x2) = labelinfo[cur_label]

        # Show the image, I
        SRP["aximg_map"].set_data(I)
        if True:
            ax = SRP["ax"][SRP["axidx_map"]]
            while len(ax.patches) > 0:
                del ax.patches[0]
            c = plt.Rectangle((int(x1),int(y1)), int(x2-x1), int(y2-y1),
                linewidth=1, edgecolor="white", facecolor='none')
            ax.add_patch(c)

        # Create P, the piece
        ax = SRP["ax"][SRP["axidx_piece"]]
        while len(ax.images) > 0:
            del ax.images[0]
        P = copy.copy(I[y1:y2+1, x1:x2+1])
        LP = label_image[y1:y2+1, x1:x2+1]
        P[LP != cur_label] = 0
        # Rotate if indicated
        P = transform.rotate(P, piece.dst_result, resize=True, mode='constant', cval=0)
        if True:
            del SRP["aximg_piece"]
            SRP["aximg_piece"] = SRP["ax"][SRP["axidx_piece"]].imshow(P)
        else:
            SRP["aximg_piece"].set_data(P)

    # Specific piece calculations
    s = SRP["text_bangle"][2].format(piece.dst_b_angle)
    SRP["wid_bangle"].set_text(s)
    s = SRP["text_angle"][2].format(piece.dst_angle)
    SRP["wid_angle"].set_text(s)

    s = SRP["text_bpol"][2].format(piece.dst_b_polarity)
    SRP["wid_bpol"].set_text(s)
    s = SRP["text_pol"][2].format(piece.dst_polarity)
    SRP["wid_pol"].set_text(s)

    s = SRP["text_result"][2].format(piece.dst_result)
    SRP["wid_result"].set_text(s)

def update_sr_rot(val):
    if ignore_sr_update:
        return
    piece = Pieces[(SRP["cur_label"], 0)]
    piece.set_b_angle(True)
    piece.set_angle(val)
    update_sr_images()

def nearest_label(new_label):
    # TBD- See if new_label is in current scope (based on show angle bool and pol
    # bool). It may be the same as current_label but Image could change
    return min(mylabels, key=lambda x : abs(x - int(round(new_label))))

def update_sr_label(val):
    #Need to set rotation bar based on piece- could change with change of label or
    #slider itself
    global ignore_sr_update
    if ignore_sr_update:
        return

    new_label = nearest_label(SRP["slider_label"].val)
    ignore_sr_update = True
    SRP["slider_label"].set_val(new_label)
    ignore_sr_update = False

    if new_label != SRP["cur_label"]:
        SRP["cur_label"] = new_label
        update_sr_images()

def update_sr():
    global I, IMX
    global aval
    if True:
        # Get a piece by label
        piece = Pieces[(SRP["cur_label"], 0)]

        # Calculate angles
        piece.set_result()

    else:
        SRP["aximg_map"].set_data(SRP["image_map"])
        SRP["image_map"] = transform.rotate(P, piece.dst_result, resize=True, mode='constant', cval=0)
        SRP["aximg_piece"].set_data(P)

def foo():
    IX = copy.copy(I)
    if val is None:
        IX = transform.rotate(IX, aval, resize=True, mode='constant', cval=0)
        IMX = ax[1].imshow(IX)
    else:
        aval = aamp.val
        IX = transform.rotate(IX, aval, resize=True, mode='constant', cval=0)
        # -----
        # Lots of garbage here, unsuccessfully trying to get around a bug in matplotlib
        # 2.0.2 where axes are not updated correctly with relim/autoscale after
        # set_data().  Can't use imshow(), because there is a memory leak with that
        if False:
            IMX.set_data(None)
            del IMX
            IMX = ax[1].imshow(IX)
        else:
            IMX.set_data(IX)
        #print ax[1].get_xbound(), ax[1].get_ybound(), ax[1].get_xlim(), ax[1].get_ylim(), IX.shape, IX.size, IMX.get_extent(), IMX.get_size()
        #IMX.set_extent(IMX.get_extent())
        if False:
            ax[1].relim()
        else:
            ax[1].axes.ignore_existing_data_limits = True
            xmin, xmax, ymin, ymax = IMX.get_extent()
            #ax[1].axes.update_datalim(
            ax[1].axes.dataLim.update_from_data_xy(((xmin, ymin), (xmax, ymax)),
                                             ax[1].axes.ignore_existing_data_limits,
                                             updatex=True, updatey=True)
            ax[1].axes.ignore_existing_data_limits = False

        ax[1].autoscale(enable=True, tight=None)
        #print ax[1].get_xbound(), ax[1].get_ybound(), ax[1].get_xlim(), ax[1].get_ylim(), IX.shape, IX.size, IMX.get_extent(), IMX.get_size()
        #plt.draw()
        # -----

def save_rotations(fake=False):
    if fake:
        # Create I, the image
        global I
        v=75
        I = data.coins()[v:-v,v:-v]

        # Create label_info, dict of pieces with labels and bounding boxes
        from skimage.filters import threshold_otsu
        from skimage.measure import label as labelit
        thresh = threshold_otsu(I)
        bw = closing(I > thresh, square(3))
        global label_image
        label_image = labelit(bw)
        properties = regionprops(label_image)
        global labelinfo
        labelinfo = {}
        for proper in properties:
            if proper.area > 100:
                labelinfo[proper.label] = proper.bbox

        # Create mylabels, a sorted list of labels for use as identifiers
        global mylabels
        mylabels = sorted(labelinfo.keys())

        # Create masks["pieces"], for use as in main
        global masks
        masks = {}
        m = copy.copy(label_image)
        m[label_image > 0] = 255
        masks["pieces"] = m

        # Create Piece objects
        foo = -10.
        for label in mylabels:
            (y1, x1, y2, x2) = labelinfo[label]
            piece = Piece(label, x1, y1, x2 - x1 + 1, y2 - y1 + 1)
            piece.set_angle(foo)
            piece.set_b_angle(True)
            Pieces[(label, 0)] = piece
            piece.set_result()
            foo += 5

    else:
        pass
    init_sr()
    if True:
        SRP["slider_label"].on_changed(update_sr_label)
        SRP["slider_rot"].on_changed(update_sr_rot)
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
        #show_top_pieces(red=True) #, blue=True, black=True)
        #get_orientation_angles("redlines")
        get_orientation_angles("bluelines")
        show_orientation_stats()
        #view_rotations()
        #save_rotations()
    except KeyboardInterrupt:
        logger.debug("KeyboardInterrupt")
    except:
        raise
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))
else:
    '''
    import matplotlib
    import numpy
    import skimage
    print matplotlib.__version__
    print numpy.__version__
    print skimage.__version__
    2.0.2
    1.14.2
    0.13.1
    '''
    logger = logmetrics.initLogger()
    t0 = logmetrics.unix_time()
    save_rotations(fake=True)
    #labelinfo = [(1,(1,2,3,4)),(2,(2,4,6,8))]
    #print sorted([l[0] for l in labelinfo])
    #mylabels = sorted([l[0] for l in labelinfo])
    #print mylabels
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))
