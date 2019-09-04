"""
Visualization module
====================
"""


from mxnet import nd

from IPython import display
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageFont, ImageDraw

import altusi.configs.config as cfg
from altusi.utils.logger import *

# colors for drawing
COLOR_RED = (0, 0, 255)
COLOR_RED_LIGHT = (49, 81, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_LIGHT_SKY_BLUE = (250, 206, 135)
COLOR_DEEP_SKY_BLUE = (255, 191, 0)
COLOR_PURPLE = (255, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def show_images(images, nrows, ncols, titles=None, scale=1.5):
    """Show images in grid view
    
    Parameters
    ----------
    images : List[array]
        List of image arrays
    nrows : int 
        Number of rows to display
    ncols : int
        Number of columns to display
    titles : Optional[List[str]]
        List of image titles
    scale : Optional[double]
        Figure scale to display
    """
    figsize = (scale * ncols, scale * nrows)
    _, axes = plt.subplots(nrows, ncols, figsize=figsize)

    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, nd.NDArray):
            img = img.asnumpy()

        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])


def plot(X, Y, title=None, xlabel=None, ylabel=None, fmts=None, legend=[],
        xlim=None, ylim=None, xscale='linear', yscale='linear',
        figsize=(6, 4)):
    """Plot 2D graph"""
    plt.rcParams['figure.figsize'] = figsize
    axes = plt.gca()
    if isinstance(X, nd.NDArray): X = X.asnumpy()
    if isinstance(Y, nd.NDArray): Y = Y.asnumpy()
    if not hasattr(X[0], '__len__'): X = [X]
    if not hasattr(Y[0], '__len__'): Y = [Y]
    if len(X) != len(Y): X = X * len(Y)
    if not fmts: fmts = ['-'] * len(X)

    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if isinstance(x, nd.NDArray): x = x.asnumpy()
        if isinstance(y, nd.NDArray): y = y.asnumpy()

        axes.plot(x, y, fmt)

    set_axes(axes, title, xlabel, ylabel, legend, xlim, ylim, xscale, yscale)


def set_axes(axes, title, xlabel, ylabel, legend, xlim, ylim, xscale, yscale):
    """Set attributes to figure"""
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    if legend: 
        axes.legend(legend)

    axes.set_xscale(xscale)
    axes.set_yscale(yscale)

    if xlim: 
        axes.set_xlim(xlim)
    if ylim: 
        axes.set_ylim(ylim)

    axes.grid()


class Animator():
    """Animator class"""

    def __init__(self, title=None, xlabel=None, ylabel=None, legend=[],
                xlim=None, ylim=None,
                xscale='linear', yscale='linear',
                fmts=None,
                nrows=1, ncols=1, figsize=(6, 4)):
        """Initialize object instance"""
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1: self.axes = [self.axes]
        
        self.config_axes = lambda : set_axes(
            self.axes[0], title, xlabel, ylabel, legend,
            xlim, ylim, xscale, yscale)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        """Add new items to animator

        Parameters
        ----------
            x, y
                items to add for visualization
        """
        if not hasattr(y, '__len__'): y = [y]
        n = len(y)
        
        if not hasattr(x, '__len__'): x = [x] * n
        
        if not self.X: self.X = [[] for _ in range(n)]
        if not self.Y: self.Y = [[] for _ in range(n)]
        if not self.fmts: self.fmts = ['-'] * n
            
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
                
        self.axes[0].cla()
        
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            
        self.config_axes()
        
        display.display(self.fig)
        display.clear_output(wait=True)
        
    def savefig(self, save_path='figure.png'):
        """Save plot
        
        Parameters
        ----------
            save_path : str
                path to save plot
        """
        self.fig.savefig(save_path)


def drawObjects(image, objects, color=COLOR_YELLOW, thickness=2):
    """Draw bounding boxes for given input objects

    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        objects : list(numpy.array([x, y, w, h] ) )
            input bounding boxes of objects to draw

    Keyword Arguments:
    ------------------
        color : tuple(B : int, G : int, R: int) (default: COLOR_YELLOW)
            drawing color for drawing
        thickness : int (default: 2)
            how thick the shape is

    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    if len(objects) == 0:
        return image
        exit()

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2) in objects:
        for i in range(thickness):
            draw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)

    del draw
    return np.asarray(image)


def drawLabels(image, objects, labels, color=COLOR_RED, thickness=2):
    """Draw bounding boxes and the corresponding labels for given input objects

    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        objects : list(numpy.array([x, y, w, h] ) )
            input bounding boxes of objects to draw
        labels  : list(list(str) )
            corresponding labels for input objects

    Keyword Arguments:
    ------------------
        color : tuple(B : int, G : int, R: int) (default: COLOR_RED)
            drawing color for drawing 
        thickness : int (default: 2)
            how thick the shape is

    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    if len(objects) == 0:
        return image
        exit()

    font = ImageFont.truetype(font=cfg.FONT, \
                    size=np.floor(3e-2 * image.shape[0] + 0.5).astype('int32') )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for (x, y, w, h), label in zip(objects, labels):
        # label_size = np.array([w, (draw.textsize(label, font) )[1] ] )
        label = '{}'.format(label)
        label_size = draw.textsize(label, font) 

        if y - label_size[1] >= 0:
            text_coor = np.array([x, y - label_size[1] ] )
        else:
            text_coor = np.array([x, y + 1] )

        for i in range(thickness):
            draw.rectangle([x+i, y+i, x+w-i, y+h-i], outline=color)
        
        draw.rectangle([tuple(text_coor), tuple(text_coor + label_size) ], fill=color)
        draw.text(text_coor, label, fill=COLOR_WHITE, font=font)
    del draw
    return np.asarray(image)


def drawInfo(image, labels, color=COLOR_RED):
    """Draw information label for an image

    Arguments:
    ----------
        image : numpy.array
            input image for drawing
        labels : list(str)
            list of label to draw

    Keyword Arguments:
    ------------------
        color : tuple(B : int, G : int, R: int) (default: COLOR_RED)
            color for drawing

    Returns:
    --------
        image : numpy.array
            output image after drawing
    """

    font = ImageFont.truetype(font=cfg.FONT, \
                size=np.floor(3e-2 * image.shape[0] + 0.5).astype('int32') )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    prv_y = 0
    for i, label in enumerate(labels):
        label_size = list(draw.textsize(label, font) )
        label_size[0] += 1
        label_size[1] += 1

        text_coor = np.array([0, prv_y] )
        prv_y += label_size[1] + 1
        label_size = tuple(label_size)

        draw.rectangle([tuple(text_coor), tuple(text_coor + label_size) ], fill=color)
        draw.text(text_coor, label, fill=COLOR_WHITE, font=font)

    del draw
    return np.asarray(image)

