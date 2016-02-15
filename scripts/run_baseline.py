#!/usr/bin/python
#
# File: run_baseline.py
# ---------------------
# Parallelize runs of a program across multiple Stanford corn machines.
# This script generates parameter combinations and hands off to
# run_baseline.exp.
#
# This script should be run from within screen on a corn server.

import os
import time

def get_server_number(counter):
  # There are 30 corn servers, named corn01 through corn31.
  return '%02d' % (counter % 30 + 1)

# Keeps track of which corn server to use.
counter = 0

for i in xrange(3):
	# Can add more parameters to the run_baseline.exp call as necessary.
	parameters = "" # Placeholder.
    os.system("/usr/bin/expect -f run_baseline.exp %s %s %s %s&" 
      % (username, password, get_server_number(counter), parameters))
    counter += 1
    time.sleep(10)
