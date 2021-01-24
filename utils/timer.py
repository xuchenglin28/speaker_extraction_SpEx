#!/usr/bin/env python

import time

class Timer(object):
    """
    A timer to record the elapsed time
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60
