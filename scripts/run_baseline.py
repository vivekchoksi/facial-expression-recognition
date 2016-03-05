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
# > # You can press "ctrl-a d" to detach from screen and "ctrl-a r" to re-attach.

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
lrs = np.random.uniform(1e-4,1e-1,3)
regs = np.random.uniform(1e-6,1e-1,3)
num_filters1 = random.sample(xrange(64,65), 1)
num_filters2 = random.sample(xrange(64,65), 1)
dropout_rates = np.random.uniform(0,0.3,1)
depth1s = random.sample(xrange(2,3), 1)
depth2s = random.sample(xrange(2,3), 1)

for d1 in depth1s:
  for d2 in depth2s:
    for nf1 in num_filters1:
      for nf2 in num_filters2:
        for lr in lrs:
          for reg in regs:
            for dr in dropout_rates:
              # These parameters will be passed to cnn-deep.py.
              parameters = "-l " + str(lr) + " -r " + str(reg) + " -d " + str(dr) + " -nf1 " + str(nf1) + " -nf2 " + str(nf2) + " -dp1 " + str(d1) + " -dp2 " + str(d2) + " -nt 3000 -e 3 -o ./"
              command = "/usr/bin/expect -f run_baseline.exp %s '%s' &" \
                % (get_server_number(counter), parameters)
              print 'Executing command:', command
              os.system(command)
          counter += 1
          time.sleep(5)
