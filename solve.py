import matplotlib.pyplot as plt
from skimage import io
from skimage import color
import logging, logmetrics

def open_source_image():
    global I
    I=io.imread('puzzle1_400dpi.tif')

    print(I[0])

def calculate_source_stats():
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

    #io.find_available_plugins() #
    #plt.ioff()
    #plt.plot(I)
    #plt.draw()
    #print plt.rcParams
    plt.rcParams['figure.figsize'] = (18,12)
    plt.imshow(I)
    #I.shape

    img_hsv = color.rgb2hsv(img)
    #F=I.astype(float)/256.0

    #print(F.size)
    #p=F[3000,990]
    #print(p)


    F=I.astype(float)[:,:2000]
    HSV=color.rgb2hsv(F)

    print(I.size)
    print(HSV.size)

if __name__ == '__main__':
    global logger
    logger = logmetrics.initLogger() #console_logging='json', file_logging='none')
    t0 = logmetrics.unix_time()
    open_source_image()
    calculate_source_stats()
    #create_mask(background)
    #create_mask(yellow)
    #create_mask(blue)
    #create_mask(red)
    #create_mask(black_ink)
    #create_mask(red_ink)
    t1 = logmetrics.unix_time()
    logger.debug(logmetrics.unix_time_elapsed(t0, t1))

