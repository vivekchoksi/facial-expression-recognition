#/usr/bin/python
#
# File: run_transfer.py
# ---------------------
# Parallelize runs of a program across multiple Stanford corn machines.
# This script generates parameter combinations and hands off to
# run_transfer.exp to run a transfer learning job.
#
# This script should be run from within screen on a corn server.
# > ssh SUNetID@corn.stanford.edu
# > cd path/to/repo/scripts
# > screen
# > python run_transfer.py
# > # You can press "ctrl-a d" to detach from screen and "screen -x" to re-attach.

import os
import time
import numpy as np
import random

def get_server_number(counter):
  # There are 30 corn servers, named corn01 through corn31.
  return '%02d' % (counter % 30 + 1)

# Keeps track of which corn server to use.
counter = 5

# Generate random parameters in range
modes = [1, 2, 3]

for mode in modes:
  # These parameters will be passed to cnn_transfer.py.
  parameters = "-m " + str(mode) + " -e 3 -o ./"
  command = "/usr/bin/expect -f run_transfer.exp %s '%s' &" \
    % (get_server_number(counter), parameters)
  print 'Executing command:', command
  os.system(command)
  counter += 1
  time.sleep(5)
