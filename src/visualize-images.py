#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm


IMG_DIM = 48

EMOTIONS = [
  'Angry',
  'Disgust',
  'Fear',
  'Happy',
  'Sad',
  'Surprise',
  'Neutral',
]


def visualize_images(width, height):
  '''
  :param width: number of columns of images to show
  :param height: number of rows of images to show
  '''
  num_images = width * height
  with open('fer2013-small.csv','r') as f:
    for i, line in enumerate(f):
      if 1 <= i <= num_images:
        label, pixels, usage = line.split(',')
        label = int(label)

        # Reformat image from array of pixels to square matrix.
        pixels = np.array([float(num) for num in pixels.split(' ')]).reshape((IMG_DIM, IMG_DIM))

        # Plot image in grayscale without axes, in the correct subplot.
        plt.subplot(height, width, i)
        plt.gca().axis('off')
        plt.imshow(pixels, cmap = cm.Greys_r)

        # Show image label as title.
        plt.title(EMOTIONS[label])

  plt.tight_layout(h_pad=0.01)
  plt.show()


def main():
  visualize_images(7, 7)

if __name__ == '__main__':
  main()