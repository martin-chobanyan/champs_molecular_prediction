#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle


class AverageKeeper(object):
    """
    Helper class to keep track of averages
    """

    def __init__(self):
        self.sum = 0
        self.n = 0
        self.running_avg = []

    def add(self, x):
        """Update the current running sum"""
        self.sum += x
        self.n += 1

    def calculate(self):
        """Calculate the current average and append to the running average"""
        avg = self.sum / self.n if self.n != 0 else 0
        self.running_avg.append(avg)
        return avg

    def reset(self, complete=False):
        """Reset the average counter
        Parameters
        ----------
        complete: bool
            If complete is True, then the running average will be reset as well
        """
        self.sum = 0
        self.n = 0
        if complete:
            self.running_avg = []


def read_pickle(path):
    """Read an existing pickle object to memory
    Parameters
    ----------
    path: str
        The file path to the pickle
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def write_pickle(obj, path):
    """Dump any object as a pickle
    Parameters
    ----------
    obj: object
        The target object to be pickled
    path: str
        The file path for the new pickle file
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
