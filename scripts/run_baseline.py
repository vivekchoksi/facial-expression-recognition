#/usr/bin/python
#
# File: run_baseline.py
# ---------------------
# Parallelize runs of a program across multiple Stanford corn machines.
# This script generates parameter combinations and hands off to
# run_baseline.exp.
#
# This script should be run from within screen on a corn server.
# > ssh SUNetID@corn.stanford.edu
# > cd path/to/repo/scripts
# > screen
# > python run_baseline.py
# > You can press "ctrl-a d" to detach from screen and "ctrl-a r" to re-attach.

import os
import time
import numpy as np
import random

def get_server_number(counter):
  # There are 30 corn servers, named corn01 through corn31.
  return '%02d' % (counter % 30 + 1)

# Keeps track of which corn server to use.
counter = 0

# Generate random parameters in range
# lrs = np.random.uniform(1e-4,1e-1,3)
# regs = np.random.uniform(1e-6,1e-1,3)
# num_filters1 = random.sample(xrange(64,65), 1)
# num_filters2 = random.sample(xrange(64,65), 1)
# dropout_rates = np.random.uniform(0,0.3,1)
# depth1s = random.sample(xrange(2,3), 1)
# depth2s = random.sample(xrange(2,3), 1)

params = [
  # # Depths 1, low dropout values, with batch norm
  # '-l 0.001 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 1 -e 10 -o ./ -save -bn',

  # # Depths 1, higher dropout values, with batch norm
  # '-l 0.001 -d1 .2 -d2 .5 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 1 -e 10 -o ./ -save -bn',

  # # Depth2 = 2, lower dropout values, without batch norm
  # '-l 0.001 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -e 10 -o ./ -save',

  # # Depth2 = 2, higher dropout values, without batch norm
  # '-l 0.001 -d1 .2 -d2 .5 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -e 10 -o ./ -save',

  # # Depth2 = 1, lower dropout values, without batch norm, with fractional max-pooling
  # '-l 0.001 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 1 -e 10 -o ./ -save -frac',

  # # Depth2 = 1, lower dropout values, with batch norm, with fractional max-pooling
  # '-l 0.001 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 1 -e 10 -o ./ -save -frac',

  # More epochs of run ID 1 with lower learning rate and loading weights.
  '-l 0.0001 -d1 0.2 -d2 .5 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -e 10 -o ./ -save ' + 
    '-w ./weights/depth2=2_fractional_pooling=False_use_batchnorm=False_dropout1=0.2_dropout2=0.5.hdf5',

  # Higher learning rate, FMP, higher dropout, and deeper net.
  '-l 0.005 -d1 0.2 -d2 .5 -r 1e-5 -nf1 32 -nf2 64 -dp1 2 -dp2 3 -e 15 -o ./ -frac -save',

  # Try best params so far with different learning rates.
  '-l 0.0005 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -e 10 -o ./ -save',

  # Try best params so far with different learning rates.
  '-l 0.0002 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -e 10 -o ./ -save',

  # Try best params so far with different learning rates.
  '-l 0.002 -d1 0 -d2 .2 -r 1e-5 -nf1 32 -nf2 64 -dp1 1 -dp2 2 -e 10 -o ./ -save',

]


for p in params:
  # These parameters will be passed to cnn_baseline.py.
  parameters = p
  command = "/usr/bin/expect -f run_baseline.exp %s '%s' &" \
    % (get_server_number(counter), parameters)
  print 'Executing command:', command
  os.system(command)
  counter += 1
  time.sleep(5)
