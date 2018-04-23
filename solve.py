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
        P = I[y1:y2+1, x1:x2+1]
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

        # consume label:bbox as dict
        d = dict(labelinfo)

        # Now for red line test ----------------------
        # get top red line pieces
        top = sorted(Pieces.values(), key=lambda Piece: Piece.src_n_rline_pix, reverse=True)
        # display top red line pieces
        if False:
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
        if False:
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
        if True:
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
    except:
        logger.debug("Encountered error")
        raise
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))

