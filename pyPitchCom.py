#!/usr/bin/python2

import sys
import wave
from math import sqrt
import numpy as np

def ReadWaveFile(filename):
    """
    Open wave file specified by filename and return data.

    Return: (waveInfo, wave_data)
      waveInfo is a dict, contains following items:
        "nchannels", "framerate", "nframes", "samplewidth"
      wave_data is a 2-dimension numpy array, the nth channel's
        data can be access by wave_data[:, n]

    Warn: This function has an assumption that the 
          wave file should be in format of 16-bit integer (short)
    """
    f = wave.open(filename, 'rb')
    waveInfo = dict()
    waveInfo["nchannels"] = f.getnchannels()
    waveInfo["framerate"] = f.getframerate()
    waveInfo["nframes"] = f.getnframes()
    waveInfo["samplewidth"] = f.getsampwidth()
    str_data = f.readframes(waveInfo["nframes"])

    # np.short is 16-bit length
    wave_data = np.fromstring(str_data, dtype=np.short)  
    wave_data = wave_data.astype(np.float16)
    wave_data /= 32768.0
    wave_data.shape = -1, waveInfo["nchannels"]
    return waveInfo, wave_data

def EnergyNorm(dataMatrix):
    """
    Normalize the energy of signals in dataMatrix.
    Arguments:
      dataMatrix:
        this is a numpy array which contains several sequences,
        dataMatrix[:, n] is the n-th sequence.
    Return:
      newDataMatrix:
        a 2-dimention numpy array after energy normalization.
    """
    nSeq = dataMatrix.shape[1]
    nLength = dataMatrix.shape[0]

    FRAME_LEN = 400
    FRAME_SHIFT = 100
    D = 2048

    WINDOW = np.hanning(FRAME_LEN).reshape(-1,1)
    nFrame = (nLength - 2 * D) / FRAME_SHIFT + 1

    newDataMatrix = np.zeros_like(dataMatrix, np.float16)

    for nthData in range(nSeq):
        for i in range(nFrame):
            framePos = i * FRAME_SHIFT - FRAME_LEN / 2 + D
            windowedWaveBuffer = dataMatrix[framePos: framePos + FRAME_LEN, nthData].reshape(-1,1) * WINDOW
            frameEnergy = sqrt(np.sum(windowedWaveBuffer * windowedWaveBuffer))

            newDataMatrix[framePos: framePos + FRAME_LEN, nthData] += (windowedWaveBuffer * WINDOW / frameEnergy).ravel()

    return newDataMatrix
            
#@profile
def AutoCorr(dataMatrix):
    """
    Compute auto-correlation function of one signal in dataMatrix
    Arguments:
      dataMatrix:
        this is a numpy array which contains several sequences,
        dataMatrix[:, n] is the n-th sequence.
    Return:
      (corr, score):
        corr is the result directly computed by auto-correlation function
        of each delta (range in 0 to CORR_N). It is a 2-dimention numpy array,
        of which the 1st dimention refers to each frame, and the 2nd refers to
        the value of delta.

        score is result after a simple procession.

    Note: This function can only process one signal in input dataMatrix.
    """
    FRAME_SHIFT = 160
    FRAME_LEN = 400
    CORR_N = 300
    CORR_S = 16
    D = 2048

    COEFF = np.linspace(float(2) - float(CORR_S)/float(CORR_N), 1, CORR_N - CORR_S + 1)
    COEFF = np.append(np.zeros(15), COEFF)

    nSeq = dataMatrix.shape[1]
    nLength = dataMatrix.shape[0]

    WINDOW = np.hanning(FRAME_LEN).reshape(-1,1)
    nFrame = (nLength - 2 * D) / FRAME_SHIFT + 1
    
    score = np.zeros((nFrame, CORR_N), np.float16)
    corr = np.zeros((nFrame, CORR_N), np.float16)

    for nthData in range(nSeq):
        for i in range(nFrame):
            framePos = (i - 1) * FRAME_SHIFT - FRAME_LEN / 2 + D
            windowedFrameBuffer = dataMatrix[framePos: framePos + FRAME_LEN, nthData].reshape(-1,1) * WINDOW

            for delta in range(CORR_N):
                corr[i][delta] = windowedFrameBuffer.ravel().dot(dataMatrix[framePos + delta: framePos + FRAME_LEN + delta].ravel())

            # Method 1: fast but has some difference with the original one
            #
            # score[i] = corr[i] * COEFF
            # tmp = (corr[i]<0)
            # score[i][tmp] = corr[i][tmp]

            # Method 2: the original method, very slow.
            #
            # for delta in range(CORR_S, CORR_N):
            #     if corr[i][delta] > 0:
            #         score[i][delta] = corr[i][delta] * (2 * CORR_N - delta) / CORR_N
            #     else:
            #         pass

    # Use no coeffs
    score = corr

    return (corr, score)


def ZeroAdding(D, dataMatrix):
    """
    Add zeros to data.
    Arguments:
      D: 
        there will be 2*D zeros to be added before the beginning
        and after the ending of data sequences.
      dataMatrix:
        this is a numpy array which contains several sequences,
        dataMatrix[:, n] is the n-th sequence.
    Return:
      newDataMatrix:
        also a numpy array after adding zeros to dataMatrix.
    """
    nSeq = dataMatrix.shape[1]
    nLength = dataMatrix.shape[0]

    data2Add = np.zeros((D, nSeq), np.float16)
    return np.concatenate((data2Add, dataMatrix, data2Add), axis=0)

def RandAdding(D, dataMatrix):
    """
    Add random values to data.
    Arguments:
      D: 
        there will be 2*D random values to be added before the beginning
        and after the ending of data sequences.
      dataMatrix:
        this is a numpy array which contains several sequences,
        dataMatrix[:, n] is the n-th sequence.
    Return:
      newDataMatrix:
        also a numpy array after adding random values to dataMatrix.
    """
    nSeq = dataMatrix.shape[1]
    nLength = dataMatrix.shape[0]

    # random value's range is [-1,1], a uniform distribution.
    data2Add = np.random.rand(D, nSeq) * 2 - 1
    dataMatrix += (np.random.rand(nLength, nSeq) - 0.5)* 0.0001
    return np.concatenate((data2Add, dataMatrix, data2Add), axis=0)

if __name__ == "__main__":
    import pylab as pl
    from pprint import pprint
    waveInfo, wave_data = ReadWaveFile(sys.argv[1])
    pprint(waveInfo)

    sig0 = RandAdding(2048, wave_data)
    sig1 = EnergyNorm(sig0)
    corr,score = AutoCorr(sig1)

    x, y = np.mgrid[:score.shape[0], 1:301]
    fig, ax = pl.subplots()
    mesh = ax.pcolormesh(x,y,score)
    pl.colorbar(mesh)
    pl.xlabel("Time(nframe)")
    pl.ylabel("Delta")
    pl.title("ACF-gram")
    pl.show()
