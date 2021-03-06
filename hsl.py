import matplotlib.pyplot as plt
#import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons
from skimage import io
from skimage import color
import logging, logmetrics
import copy

def open_rgb_image():
    global I
    #I=plt.imread('hsl.tif')
    I1=plt.imread('puzzle1_400dpi.tif')
    #I=I1[500:1000,:700,:]
    #I=I1[2800:3400,1600:2100,:]
    #I=I1[2850:3550,1200:2000,:]
    I=I1[3100:3600,1700:3300,:]

def convert_source_to_hsl():
    global H
    F=I.astype(float)/256.0
    H=color.rgb2hsv(F)

def init_hsl_params():
    global ph, ps, pl
    global phd, psd, pld
    ph = 0.15
    ps = 0.8
    pl = 0.9
    phd = 0.15
    psd = 0.15
    pld = 0.15
    (ph,ps,pl,phd,psd,pld) = (0.18, 0.80, 0.62, 0.02, 0.31, 0.15)
    (ph,ps,pl,phd,psd,pld) = (0.01, 0.01, 0.01, 0.10, 0.20, 0.40)
    (ph,ps,pl,phd,psd,pld) = (0.37, 0.21, 0.35, 0.33, 1.00, 1.00)
    (ph,ps,pl,phd,psd,pld) = (0.92, 1.00, 1.00, 0.10, 0.70, 0.30)
    (ph,ps,pl,phd,psd,pld) = (0.52, 0.90, 0.90, 0.48, 0.48, 0.48)
    (ph,ps,pl,phd,psd,pld) = (0.40, 0.50, 0.50, 0.401, 0.501, 0.501)
    (ph,ps,pl,phd,psd,pld) = (0.10, 0.98, 0.93, 0.02, 0.15, 0.01)

def create_hsl_mask():
    global mask
    HL = ph - phd
    HU = ph + phd
    if HL < 0.0:
        HL2 = HL + 1.0
    else:
        HL2 = 1.0
    if HU > 1.0:
        HU2 = HU - 1.0
    else:
        HU2 = 0.0
    SL = ps - psd
    SU = ps + psd
    LL = pl - pld
    LU = pl + pld
    #print HL, HU, SL, SU, LL, LU, HL2, HU2
    mask = \
        (((H[:,:,0] > HL) & (H[:,:,0] < HU)) |
            (H[:,:,0] > HL2) |
            (H[:,:,0] < HU2)) & \
        (H[:,:,1] > (SL)) & \
        (H[:,:,1] < (SU)) & \
        (H[:,:,2] > (LL)) & \
        (H[:,:,2] < (LU))
    #print(type(mask),mask.dtype)

FIX_LEAK=True

def update(val):
    global ph, ps, pl
    global phd, psd, pld
    global fig, ax_list
    global I1, I2, I3, I4
    global IM0, IM1, IM2, IM3, IM4
    global hamp, samp, lamp, hdamp, sdamp, ldamp
    global mask
    ph = hamp.val
    ps = samp.val
    pl = lamp.val
    phd = hdamp.val
    psd = sdamp.val
    pld = ldamp.val
    create_hsl_mask()

    if FIX_LEAK:
        I1 = copy.copy(I)
        I1[mask] = 0

        I2 = copy.copy(I)
        I2[~(mask)] = 0

        I3 = copy.copy(H)
        I3[mask] = 0

        I4 = copy.copy(H)
        I4[~(mask)] = 0

        if val is None:
            ax_list[0,0].imshow(I)
            IM1 = ax_list[0,1].imshow(I1)
            IM2 = ax_list[0,2].imshow(I2)
            IM0 = ax_list[0,3].imshow(mask)
            ax_list[1,0].imshow(H)
            IM3 = ax_list[1,1].imshow(I3)
            IM4 = ax_list[1,2].imshow(I4)
        else:
            IM0.set_data(mask)
            IM1.set_data(I1)
            IM2.set_data(I2)
            IM3.set_data(I3)
            IM4.set_data(I4)
    else:
        ax_list[0,3].imshow(mask)

        # Note: to prevent memory leak, perform imshow once and save the result
        # then instead of imshow() here, use im.set_data()
        I1 = copy.copy(I)
        I1[mask] = 0
        ax_list[0,1].imshow(I1)

        I2 = copy.copy(I)
        I2[~(mask)] = 0
        ax_list[0,2].imshow(I2)

        I3 = copy.copy(H)
        I3[mask] = 0
        ax_list[1,1].imshow(I3)

        I4 = copy.copy(H)
        I4[~(mask)] = 0
        ax_list[1,2].imshow(I4)

if __name__ == "__main__":
    #print(matplotlib.__version__)
    global fig, ax_list
    global I1, I2, I3, I4
    global hamp, samp, lamp, hdamp, sdamp, ldamp
    global mask
    open_rgb_image()
    convert_source_to_hsl()
    init_hsl_params()
    create_hsl_mask()

    fig, ax_list = plt.subplots(3,4,figsize=(14,7))
    ax_list[1, 3].axis('off')
    ax_list[2, 0].axis('off')
    ax_list[2, 1].axis('off')
    ax_list[2, 2].axis('off')
    ax_list[2, 3].axis('off')
    plt.subplots_adjust(
        left = 0.05,
        right = 0.95,
        bottom = 0.1,
        top = 0.95,
        wspace = 0.2,
        hspace = 0.2)

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

    if FIX_LEAK:
        update(None)
    hamp.on_changed(update)
    samp.on_changed(update)
    lamp.on_changed(update)
    hdamp.on_changed(update)
    sdamp.on_changed(update)
    ldamp.on_changed(update)

    if FIX_LEAK:
        pass
    else:
        ax_list[0,0].imshow(I)
        ax_list[1,0].imshow(H)
        update(None)
    plt.show()

