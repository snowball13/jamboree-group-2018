import csv
import numpy as np
import matplotlib.pyplot as plt


def get_prediction_data():
    ### GNL need to remove 12043 - entry 2041

    path = 'AIR_data/Predictor_Data/'

    # with open(path+'Global_NightLights.csv', 'r') as f:
    #   reader = csv.reader(f)
    #   GNL = list(reader)
    #
    # GNL = np.array(GNL)
    # GNL = GNL[1:np.size(GNL)+1,:]
    # print(np.size(GNL))
    # GNLtreat = np.zeros([2369,2])#np.delete(GNL, 1, 2041)
    # GNLtreat[0:2040,:] = GNL[0:2040,:]
    # GNLtreat[2041:-1,:] = GNL[2042:-1,:]

    with open(path + 'GlobalPopCounts.csv', 'r') as f:
        reader = csv.reader(f)
        GPC = list(reader)

    GPC = np.array(GPC)
    GPC = GPC[1:np.size(GPC) + 1, :]

    with open(path + 'GlobalGDP_Stats.csv', 'r') as f:
        reader = csv.reader(f)
        GGDP = list(reader)

    GGDP = np.array(GGDP)
    GGDP = GGDP[1:np.size(GGDP) + 1, :]

    with open(path + 'GEM_HistoricalFreq.csv', 'r') as f:
        reader = csv.reader(f)
        GGHF = list(reader)

    GGHF = np.array(GGHF)
    GGHF = GGHF[1:np.size(GGHF) + 1, :]

    # with open(path+'Global475yrPGA.csv', 'r') as f:
    #   reader = csv.reader(f)
    #   GPGA = list(reader)
    #
    # GPGA = np.array(GPGA)
    # GPGA = GPGA[1:np.size(GPGA)+1,:]

    with open(path + 'GlobalAverageSoil.csv', 'r') as f:
        reader = csv.reader(f)
        GAS = list(reader)

    GAS = np.array(GAS)
    GAS = GAS[1:np.size(GAS) + 1, :]

    # with open(path+'GlobalMIPC_Stats.csv', 'r') as f:
    #   reader = csv.reader(f)
    #   GMIPC = list(reader)
    #
    # GMIPC = np.array(GMIPC)
    # GMIPC = GMIPC[1:np.size(GMIPC)+1,:]

    # with open(path+'GlobalSeismicBudget.csv', 'r') as f:
    #   reader = csv.reader(f)
    #   GSB = list(reader)
    #
    # GSB = np.array(GSB)
    # GSB = GSB[1:np.size(GSB)+1,:]

    # with open(path+'USGS_HistoricalFreq.csv', 'r') as f:
    #   reader = csv.reader(f)
    #   UHF = list(reader)
    #
    # UHF = np.array(UHF)
    # UHF = UHF[1:np.size(UHF)+1,:]

    # print('Size of data sets')
    # print(len(GNL))
    # print(len(GPC))
    # print(len(GGDP))
    # print(len(GGHF))
    # print(len(GPGA))
    # print(len(GAS))
    # print(len(GMIPC))
    # print(len(GSB))
    # print(len(UHF))

    # print(GPC[0,:])
    # print(GPC[:,0])

    ############# This is how you find the observations that are common between all predictors

    int1 = np.array(sorted(set(GPC[:, 0]).intersection(GGDP[:, 0])))
    int2 = np.array(sorted(set(GGHF[:, 0]).intersection(GAS[:, 0])))
    # int1 = np.array(list(set(GPC[:,0]).intersection(GGDP[:,0])))
    # int2 = np.array(list(set(GGHF[:,0]).intersection(GPGA[:,0])))
    # int3 = np.array(list(set(GAS[:,0]).intersection(GMIPC[:,0])))
    # int4 = np.array(list(set(GSB[:,0]).intersection(UHF[:,0])))
    inttotal = np.array(sorted(set(int1).intersection(np.array(sorted(set(int2))))))
    # inttotal = np.array(list(set(int1).intersection(np.array(list(set(int2).intersection(np.array(list(set(int3).intersection(np.array(list(set(int4).intersection(GNL[:,0]))))))))))))
    # print len(inttotal)\n",
    # print max([len(ObIdorig31), len(ObIdunpre31), len(ObId1500_31),len(ObId1000_31),len(ObId500_31),len(ObId67_31)])\n",

    print(len(inttotal))

    ############# Find the indices for each of the predictors

    GPCindex = np.nonzero(np.in1d(GPC[:, 0], inttotal))[0]
    GGDPindex = np.nonzero(np.in1d(GGDP[:, 0], inttotal))[0]
    GGHFindex = np.nonzero(np.in1d(GGHF[:, 0], inttotal))[0]
    # GPGAindex = np.nonzero(np.in1d(GPGA[:,0], inttotal))[0]
    GASindex = np.nonzero(np.in1d(GAS[:, 0], inttotal))[0]
    # GMIPCindex = np.nonzero(np.in1d(GMIPC[:,0], inttotal))[0]
    # GSBindex = np.nonzero(np.in1d(GSB[:,0], inttotal))[0]
    # UHFindex = np.nonzero(np.in1d(UHF[:,0], inttotal))[0]
    # GNLindex = np.nonzero(np.in1d(GNL[:,0], inttotal))[0]



    return GGHF[GGHFindex, 0].astype(np.float64), GPC[GPCindex, 1].astype(np.float64), GGDP[GGDPindex, 1].astype(
        np.float64), GGHF[GGHFindex, 1].astype(np.float64), GAS[GASindex, 1].astype(np.float64)
