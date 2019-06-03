# bbwidgets
Interactive Widgets for the Jupyter Notebook

## Installation

Install the current release of bbwidgets from git using pip.

`pip install git+https://github.com/BodenmillerGroup/bbwidgets`

Optional dependencies include the `tifffile` and `imageio` packages.

## Usage

In your Jupyter notebook. enable the `matplotlib inline` backend and import bbwidgets.

```python3
%matplotlib inline
import bbwidgets
```

### MultichannelView

The `MultichannelView` widget enables the quick visualization of multi-channel images within your Jupyter notebook environment. While it does NOT aim to be a full-fledged image browser, it supports basic functions such as interactive histogram normalization, channel coloring and zoomed/tile view. The main purpose of the MultichannelView widget is to allow the interactive inspection of image processing pipelines.

```
> help(bbwidgets.view_multichannel)
Help on function view_multichannel in module bbwidgets.multichannel_view:

view_multichannel(images, image_names=None, channel_names=None, channel_vmins=None, channel_vmaxs=None,
    channel_states=None, channel_colors=None, num_histogram_bins=100, selected_channel=0, num_columns=4,
    backend='tifffile')

    Interactively visualize a list of multi-channel images
    
    Returns a MultichannelView instance, which can be displayed directly in Jupyter notebooks
    
    images: list of images; either numpy arrays of shape (c, y, x), or paths to image files on disk
    image_names: list of image names; defaults to file names if None and images is a list of file paths
    channel_names: list of channel names
    channel_vmins: list of channel minima; values below these thresholds will be clipped
    channel_vmaxs: list of channel maxima; values above these thresholds will be clipped
    channel_states: list of booleans (True, False) indicating the state (active, inactive) of each channel
    channel_colors: list of channel colors (matplotlib-supported color strings)
    num_histogram_bins: number of bins to use for displaying the channel histogram
    selected_channel: index of the selected channel
    num_columns: number of columns in the "Gallery" and "Tiles" views
    backend: backend ('tifffile', 'imageio') to use for reading image files if images is a list of strings

```

Usage example:

```python3
%matplotlib inline
from bbwidgets import view_multichannel
from tifffile import imread

img1 = imread('/path/to/img1.ome.tiff')
img2 = imread('/path/to/img2.ome.tiff')
images = [img1, img2]

# alternatively, use file paths to specify the images:
# images = ['/path/to/img1.ome.tiff', '/path/to/img2.ome.tiff']

# in this example, images have 3 channels
channel_names = ['CD44', 'SMA', 'Iridium']

view_multichannel(images, channel_names=channel_names, num_columns=2)
```

## License

This project is licensed under the MIT license.

See the [LICENSE](LICENSE) file in this repository for details.