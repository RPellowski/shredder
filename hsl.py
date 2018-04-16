import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from skimage import color
import logging, logmetrics

def open_rgb_image():
    global I
    #I=io.imread('hsl.tif')
    #print(I[0])
    I=plt.imread('hsl.tif')

def convert_source_to_hsl():
    global H
    F=I.astype(float)/256.0
    H=color.rgb2hsv(F)

def init_hsl_params():
    global ph, ps, pl
    global phd, psd, pld
    ph = 0.66
    ps = 0.5
    pl = 0.5
    phd = 0.1
    psd = 0.1
    pld = 0.1

function create_hsl_mask():
    pass

def bar():
    plt.imshow(H)
    plt.show()

def foo():
    return "hello"

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)


from matplotlib.widgets import Slider, Button, RadioButtons
'''
plots
  rgb
  hsl
  rgb and mask
  hsl and mask
  rgb and not mask
  hsl and not mask
  mask
views
  x,y
  rgb 0-255, 0-1
  hsl 0-255, 0-1
controls
  h 0-255, 0-1
  h range 0-255, 0-1
  s 0-255, 0-1
  s range 0-255, 0-1
  l 0-255, 0-1
  l range 0-255, 0-1

'''
def update(val):
    ph = 0.66
    ps = 0.5
    pl = 0.5
    phd = 0.1
    psd = 0.1
    pld = 0.1

def baz():
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()

if __name__ == "__main__":
    open_rgb_image()
    convert_source_to_hsl()
    init_hsl_params()
    #print(matplotlib.__version__)
    done = False
    for i in [1]:
        plt.figure(figsize=(11,6))
        #display_rgb_image
        plt.subplot(341)
        plt.imshow(I)
        #display_rgb_image and mask
        plt.subplot(342)
        plt.imshow(I)
        #display_rgb_image and not mask
        plt.subplot(343)
        plt.imshow(I)

        #display_hsl_image
        plt.subplot(345)
        plt.imshow(H)
        #display_hsl_image and mask
        plt.subplot(346)
        plt.imshow(H)
        #display_hsl_image and not mask
        plt.subplot(347)
        plt.imshow(H)

        #display_mask
        mask = H
        plt.subplot(344)
        plt.imshow(mask)

        #display_interactive_params
        #linebuilder = LineBuilder(line)
        #plt.subplot(2,2,4)
        #ax = plt.gca()
        #plt.gcf().canvas.mpl_connect('pick_event', onpick)

        axcolor = 'lightgoldenrodyellow'
        hbar  = plt.axes([0.15, 0.30, 0.55, 0.03], facecolor=axcolor)
        sbar  = plt.axes([0.15, 0.25, 0.55, 0.03], facecolor=axcolor)
        lbar  = plt.axes([0.15, 0.20, 0.55, 0.03], facecolor=axcolor)

        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        hdbar = plt.axes([0.15, 0.15, 0.55, 0.03], facecolor=axcolor)
        sdbar = plt.axes([0.15, 0.10, 0.55, 0.03], facecolor=axcolor)
        ldbar = plt.axes([0.15, 0.05, 0.55, 0.03], facecolor=axcolor)

        hamp = Slider(hbar, 'Hue', 0., 1.0, valinit=ph)
        samp = Slider(sbar, 'Sat', 0., 1.0, valinit=ps)
        lamp = Slider(lbar, 'Lum', 0., 1.0, valinit=pl)
        hdamp = Slider(hdbar, 'Hue delta', 0., 1.0, valinit=phd)
        sdamp = Slider(sdbar, 'Sat delta', 0., 1.0, valinit=psd)
        ldamp = Slider(ldbar, 'Lum delta', 0., 1.0, valinit=pld)

        hamp.on_changed(update)
        samp.on_changed(update)
        lamp.on_changed(update)
        hdamp.on_changed(update)
        sdamp.on_changed(update)
        ldamp.on_changed(update)

        plt.show()

