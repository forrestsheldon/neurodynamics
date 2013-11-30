"""
Functions for finding attributes of spikes, such as the interspike interval and
phase.

Written by Jeffrey Bush (jeff@coderforlife.com) 2010
"""

import scipy as sp

def isi(t, V, spike_thresh=0):
    """isi_mean, isi_dev = isi(t, V, spike_thresh=0)

    Given voltage (V) and time (t) vectors, isi calculates the mean interspike
    interval (isi_mean) and the standard deviation of the interspike interval
    (isi_dev).

    You can optionally specify the spike threshold (defaults to 0).

    This uses an assumption that every time it spikes, the voltage increases
    above the given spike threshold.

    The method used here is not robust. If the data is noisy then there will
    likely be false positives. With a model however, it should work very
    well.
    """
    time = t[sp.logical_and(V[:-1] < spike_thresh, V[1:] >= spike_thresh)]
    dt = sp.diff(time)
    return sp.mean(dt), sp.std(dt)

def spk_phase(t, V1, V2, spike_thresh=0):
    """phase_mean, isi_mean = spk_phase(t, V1, V2, spike_thresh=0)
    
    Given two voltage vectors (V1 and V2) and time vector (t), phase calculates
    the mean phase of the spikes in radians (phase_mean) and the mean
    interspike interval (isi_mean).

    You can optionally specify the spike threshold (defaults to 0).

    This uses an assumption that every time it spikes, the voltage increases
    above the given spike threshold.

    The method used here is not robust. If the data is noisy then there will
    likely be false positives. With a model however, it should work very
    well.
    """
    
    time1 = t[sp.logical_and(V1[:-1] < spike_thresh, V1[1:] >= spike_thresh)]
    time2 = t[sp.logical_and(V2[:-1] < spike_thresh, V2[1:] >= spike_thresh)]

    l = sp.amin([len(time1), len(time2)])
    isi_mean = sp.mean(sp.diff(time1))
    phase_mean = sp.mean((time1[0:l]-time2[0:l]) / isi_mean * 2 * sp.pi)
    return phase_mean, isi_mean
