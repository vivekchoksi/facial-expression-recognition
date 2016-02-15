#!/usr/bin/python
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
# > # You can press ctrl-d to detach from screen and ctrl-r to re-attach.

import os
import time

def get_server_number(counter):
  # There are 30 corn servers, named corn01 through corn31.
  return '%02d' % (counter % 30 + 1)

# Keeps track of which corn server to use.
counter = 0

# TODO(arushir): Modify this for loop to loop over appropriate parameters.
# These parameters will be passed to cnn-baseline.py.
for parameters in ['...', '...', '...']:
  os.system("/usr/bin/expect -f run_baseline.exp %s %s &"
    % (get_server_number(counter), parameters))
  counter += 1
  time.sleep(5)
