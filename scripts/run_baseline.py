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
# > # You can press "ctrl-a d" to detach from screen and "ctrl-a r" to re-attach.

import os
import time

def get_server_number(counter):
  # There are 30 corn servers, named corn01 through corn31.
  return '%02d' % (counter % 30 + 1)

# Keeps track of which corn server to use.
counter = 0

# These parameters will be passed to cnn-baseline.py.
for parameters in ['-l 1e-8', '-l 1e-7', '-l 1e-6', '-l 1e-5', '-l 1e-4', '-l 1e-3', '-l 1e-2', '-l 1e-1']:
  command = "/usr/bin/expect -f run_baseline.exp %s '%s' &" \
    % (get_server_number(counter), parameters)
  print 'Executing command:', command
  os.system(command)
  counter += 1
  time.sleep(5)
