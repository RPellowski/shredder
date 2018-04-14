# Purpose
A place to hold ideas that might be developed into algorithms or to help solve in other ways

## Knowns
```
color (yellow, blue, red, black)
lines
 horizontal (blue)
 vertical (red)
 none (margin)
orientation (calculated from lines)
polarity (up/down not determined by lines)
markings (handwriting)
prototype (blank paper)
 zones (x,y areas)
  straight edges (borders)
  lines (horizontal/vertical)
  colors
   handwritten area centralized
   background color variance across the page
```
 
## Unknowns TBD
```
origin location (input data)
 x, y
orientation
 from horizontal/vertical lines
 from edges
edges
 from orientation
 account for polarity
 try to make unique matches
polarity
destination
 zone
 location x, y
displays/domains
 graphics/pictorial
 statistics
conversion between domains
tools
 graphics
 data measurement, creation and conversion
 machine learning
 feedback into algorithm
 statistics
  individual piece
   volume, color
  algorithmic
   progress
   resources consumed
    time, cpu
```
## Algorithms to consider

* Chan-Vese Segmentation
* Edge operators
* Canny edge detector
* Blob Detection
* Using geometric transformations

## Outcome
```
 every piece has a destination location
 graphical display of completed solution
```

## Approach
```
while not done
 calculate stats
 save state
 show in multiple domains
 take some action
```

## Refs
* [interactive html5 canvas](https://github.com/simonsarris/Canvas-tutorials/blob/master/shapes.js)
* [https://www.pygame.org/docs/tut/PygameIntro.html](https://www.pygame.org/docs/tut/PygameIntro.html)
* [https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy](https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy)
* [https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images/18461475](https://stackoverflow.com/questions/18446804/python-read-and-write-tiff-16-bit-three-channel-colour-images/18461475)
* [https://gis.stackexchange.com/questions/76919/is-it-possible-to-open-rasters-as-array-in-numpy-without-using-another-library](https://gis.stackexchange.com/questions/76919/is-it-possible-to-open-rasters-as-array-in-numpy-without-using-another-library)
* [http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py](http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py)
* [http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py)
* [https://youtu.be/5e9jhgiqbzc](https://youtu.be/5e9jhgiqbzc)
* [https://github.com/scikit-image/skimage-tutorials](https://github.com/scikit-image/skimage-tutorials)

* [https://docs.python.org/3/library/logging.handlers.html](https://docs.python.org/3/library/logging.handlers.html)
* [https://docs.python.org/3/howto/logging-cookbook.html#using-file-rotation](https://docs.python.org/3/howto/logging-cookbook.html#using-file-rotation)
* [http://docs.python-guide.org/en/latest/writing/logging/](http://docs.python-guide.org/en/latest/writing/logging/)

