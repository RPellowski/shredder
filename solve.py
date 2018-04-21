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
    #plt.rcParams['figure.figsize'] = (14,8)
    #plt.imshow(I)
    #plt.show()

def calculate_source_stats():
    logger.info(
        "Image shape: {} Image size: {} Image datatype: {} Area: {} sq in".format(
        I.shape, I.size, I.dtype, float(I.shape[0] * I.shape[1])/DPI/DPI)
        )

def create_mask(mask):
    fname = "mask_" + mask + '.npy'
    if os.path.isfile(fname):
        logger.info("Reading mask {}".format(fname))
        m = np.load(fname)
    else:
        logger.info("Creating mask {}".format(fname))
        m = np.zeros(I.shape[:-1], dtype=bool)
        #for x in range(7):
        #print(m.shape,m.size,m.dtype)
        #print(sys.getsizeof(ms))
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
        #print(mask, ph,ps,pl,phd,psd,pld)
        #print(HL,HU,SL,SU,LL,LU)

        # Have to do it line by line due to memory limitations
        for y in range(I.shape[0]):
            iline = img_as_float(I[y:y+1,:,:])
            ihsv = color.rgb2hsv(iline)
            m[y] = \
                (((ihsv[:,:,0] >= HL) & (ihsv[:,:,0] <= HU)) |
                    (ihsv[:,:,0] >= HL2) |
                    (ihsv[:,:,0] <= HU2)) & \
                (ihsv[:,:,1] >= SL) & \
                (ihsv[:,:,1] <= SU) & \
                (ihsv[:,:,2] >= LL) & \
                (ihsv[:,:,2] <= LU)
            if mask=="pieces":
                if y < 500:
                    m[y,:1300] = False
        percentage = np.count_nonzero(m)*100.0/m.size
        logger.debug({mask:str(percentage)})
        np.save(fname, m)
    return m

def label_blobs():
    from skimage.morphology import closing, square
    #from skimage.segmentation import clear_border
    from skimage.measure import label, regionprops
    #from skimage.color import label2rgb
    #from skimage.feature import blob_dog, blob_log, blob_doh

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
            #print proper.area, proper.bbox, proper.centroid, len(proper.coords)
            labelinfo.append((proper.label, proper.bbox))
    logger.info("Labeled pieces: {}".format(pieces))
    '''
    fig, ax = plt.subplots(1,figsize=(14,7))
    ax.imshow(I)
    for _,bbox in labelinfo:
        (y1, x1, y2, x2) = bbox
        c = plt.Rectangle((int(x1),int(y1)), int(x2-x1), int(y2-y1),
            linewidth=1, edgecolor="white", facecolor='none')
        ax.add_patch(c)
    plt.show()
    '''
    return (label_image, labelinfo)

def foo():
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    #http://www.scipy-lectures.org/packages/scikit-image/index.html
    # scipy.ndimage.find_objects() is useful to return slices on object in an image.
    # See also for some properties, functions are available as well in
    # scipy.ndimage.measurements with a different API (a list is returned).
    # https://www.datasciencecentral.com/profiles/blogs/interactive-image-segmentation-with-graph-cut-in-python
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
    # https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation
    pass

if __name__ == '__main__':
    #io.find_available_plugins() #
    global logger
    global masks
    logger = logmetrics.initLogger() #console_logging='json', file_logging='none')
    t0 = logmetrics.unix_time()
    try:
        open_source_image()
        calculate_source_stats()
        masks = {}
        # If we need more accuracy, redo masks so that pieces = paper + colors
        for mask in HSL_PARAMS.keys():
            masks[mask] = create_mask(mask)
        (label_image, labelinfo) = label_blobs()
        for linf in labelinfo[0]:
            label, bbox = linf
            (y1, x1, y2, x2) = bbox
            I1 = I[y1:y2+1,x1:x2+1]
            L1 = label_image[y1:y2+1,x1:x2+1]
            I1[L1!=label] = 0
            fname = "src_" + str(label) + '.npy'
            if not os.path.isfile(fname):
                print "creating", fname
                np.save(fname, I1)
            mname = "meta_" + str(label) + '.npy'

        #np.set_printoptions(threshold=100000, linewidth=320)
        #with open("Output.txt", "w") as text_file:
        #    text_file.write(str(I1))
        #    text_file.write(str(L1))

        #fig, ax = plt.subplots(1,figsize=(14,7))
        #ax.imshow(I1)
        #plt.show()
    except:
        logger.debug("Encountered error")
        raise
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))

