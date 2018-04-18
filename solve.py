import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import color
from skimage import img_as_float
import logging, logmetrics
from piece import *
import copy
import os

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
    fname = mask + '.npy'
    if os.path.isfile(fname):
        print("Reading", fname)
        m = np.load(fname)
    else:
        print("Creating", fname)
        m = np.zeros(I.shape[:-1], dtype=bool)
        #for x in range(7):
        #print(m.shape,m.size,m.dtype)
        #print(sys.getsizeof(ms))
        HL = HSL_PARAMS[mask][0] - HSL_PARAMS[mask][3]
        HU = HSL_PARAMS[mask][0] + HSL_PARAMS[mask][3]
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
                (ihsv[:,:,0] >= HL) & \
                (ihsv[:,:,0] <= HU) & \
                (ihsv[:,:,1] >= SL) & \
                (ihsv[:,:,1] <= SU) & \
                (ihsv[:,:,2] >= LL) & \
                (ihsv[:,:,2] <= LU)
            if mask=="pieces":
                if y < 500:
                    m[y,:1300] = True
        percentage = np.count_nonzero(m)*100.0/m.size
        logger.debug({mask:str(percentage)})
        np.save(fname, m)
    return m

def label_blobs():
    pass

def foo():
    #http://www.scipy-lectures.org/packages/scikit-image/index.html
    #from skimage import measure
    #properties = measure.regionprops(labels_rw)
    #[prop.area for prop in properties]

    #[prop.perimeter for prop in properties]

    # blobs_labels = measure.label(blobs, background=0)
    # scipy.ndimage.find_objects() is useful to return slices on object in an image.
    # See also for some properties, functions are available as well in
    # scipy.ndimage.measurements with a different API (a list is returned).

    # https://www.datasciencecentral.com/profiles/blogs/interactive-image-segmentation-with-graph-cut-in-python
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
    # https://www.toptal.com/python/computational-geometry-in-python-from-theory-to-implementation
    pass

from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.feature import blob_dog, blob_log, blob_doh

if __name__ == '__main__':
    #io.find_available_plugins() #
    global logger
    global masks
    logger = logmetrics.initLogger() #console_logging='json', file_logging='none')
    t0 = logmetrics.unix_time()
    open_source_image()
    calculate_source_stats()
    masks = {}
    # If we need more accuracy, redo masks so that pieces = paper + colors
    for mask in ['pieces']: #HSL_PARAMS.keys():
        masks[mask] = create_mask(mask)
    #plt.rcParams['figure.figsize'] = (14,8)
    fig, ax = plt.subplots(1,figsize=(14,8), sharex=True, sharey=True)
    I1 = copy.copy(I)
    I1[masks["pieces"]] = 0
    I1[~masks["pieces"]] = 255
    gray = I1[:,:,0] #[500:1500,:1000,0]
    bw = closing(gray > 128, square(3))
    #cleared = clear_border(bw)
    label_image,num = label(bw,return_num=True)
    print num
    properties = regionprops(label_image)
    areas = 0
    blobs = []
    for proper in properties:
        if proper.area > 100:
            areas += 1
            #print proper.area, proper.bbox, proper.centroid
            blobs.append(proper)
    print "areas:", areas
    #for image in label_image:
    #   print image
    #image_label_overlay = label2rgb(label_image, image=gray)
    ax.imshow(I1)
    #blobs = blob_doh(gray, min_sigma=30, max_sigma=300, threshold=.01)
    for blob in blobs:
        #y, x, r = blob
        (y1,x1,y2,x2) = blob.bbox
        #c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        #c = plt.Rectangle((int(x - r),int(y - r)), int(2*r), int(2*r),linewidth=2,edgecolor="red",facecolor='none')
        c = plt.Rectangle((int(x1),int(y1)),int(x2-x1),int(y2-y1),linewidth=2,edgecolor="red",facecolor='none')
        ax.add_patch(c)
    plt.show()

    label_blobs()
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))

