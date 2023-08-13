# %%

# https://stackoverflow.com/questions/44889508/remove-highly-correlated-columns-from-a-pandas-dataframe

# 
# Ladataan käytettävät paketit:
import numpy as np
import pandas as pd

from copy import copy, deepcopy

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def _findCorrelation(x, cutoff = 0.9, verbose = False):

    removed = []

    # Irroitetaan arvot Numpy-taulukkoon
    _x = x.values

    # Napataan rivien lkm muistiin
    varnum = _x.shape[0]

    # Varmistetaan, että matriisi symmetrinen
    if check_symmetric(_x) == False:
        print("Ei oo symmetrinen")
        return None
    
    # Mitä tämä tarkoittaa?
    if varnum == 1:
        print("only one variable given")
        return None
    
    # Jatketaan lukujen absoluuttisilla arvoilla
    _x = np.abs(_x)

    #   originalOrder <- 1:varnum
    originalOrder = np.arange(0, varnum)

    # Creates deep copy of object
    tmp = deepcopy(_x)

    # tyhjennetään diagonaaliakseli
    # - oletus on, että korrelaatiomatriisi sisältää float-arvoja....
    # - na.arvoja ei huomioda keskiarvoja laskettaessa
    np.fill_diagonal(tmp, np.nan)

    # Muuttujien järjestys korrelaatioarvojen keskiarvon perustella
    maxAbsCorOrder = np.nanmean(tmp, axis = 0).argsort()[::-1]

    for i in range(len(maxAbsCorOrder) - 1):
        iIndx = maxAbsCorOrder[i]

        # Onko sarake jo aiemmin poistettu
        if iIndx in removed:
            continue

        for j in range(i + 1, len(maxAbsCorOrder)):

            jIndx = maxAbsCorOrder[j]

            # Onko jompikumpi jo poistettu
            if iIndx in removed or jIndx in removed:
                continue
            
            # Ylittääkö muuttujien välinen korrelaatio raja-arvon
            if tmp[iIndx][jIndx] > cutoff:
                    
                mn1 = np.nanmean(tmp[iIndx,:])
                mn2 = np.nanmean(tmp[:,jIndx])

                if mn1 > mn2:

                    removed.append(iIndx)
                    tmp[iIndx,:] = np.nan
                    tmp[:,iIndx] = np.nan
                else:

                    removed.append(jIndx)
                    tmp[jIndx,:] = np.nan
                    tmp[:,jIndx] = np.nan


    return removed



def findCorrelation(x, cutoff = 0.9, verbose = False):

    removed = []

    # Varmistetaan, että aineiston tietotyypit ovat float tyyppisiä
    if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

    # Irroitetaan arvot Numpy-taulukkoon
    _x = x.values

    # Napataan rivien lkm muistiin
    varnum = _x.shape[0]

    # Varmistetaan, että matriisi symmetrinen
    if check_symmetric(_x) == False:
        print("Ei oo symmetrinen")
        return None
    
    # Mitä tämä tarkoittaa?
    if varnum == 1:
        print("only one variable given")
        return None
    
    # Jatketaan lukujen absoluuttisilla arvoilla
    _x = np.abs(_x)


    # tyhjennetään diagonaaliakseli
    # - oletus on, että korrelaatiomatriisi sisältää float-arvoja....
    # - na.arvoja ei huomioda keskiarvoja laskettaessa
    np.fill_diagonal(_x, np.nan)

    # Muuttujien järjestys korrelaatioarvojen keskiarvon perustella
    maxAbsCorOrder = np.nanmean(_x, axis = 0).argsort()[::-1]

    for i in range(len(maxAbsCorOrder) - 1):
        iIndx = maxAbsCorOrder[i]

        # Ota jotenkin käyttöön...
        # if np.any(_x > cutoff) == False:
        #    print("Ei enää", i)
        # else:
        #    print(".", sum(_x > cutoff))

        # Onko sarake jo aiemmin poistettu
        if iIndx in removed:
            continue

        for j in range(i + 1, len(maxAbsCorOrder)):

            jIndx = maxAbsCorOrder[j]

            # Onko jompikumpi jo poistettu
            if iIndx in removed or jIndx in removed:
                continue
            
            # Ylittääkö muuttujien välinen korrelaatio raja-arvon
            if _x[iIndx][jIndx] > cutoff:
                    
                mn1 = np.nanmean(_x[iIndx,:])
                mn2 = np.nanmean(_x[:,jIndx])

                if mn1 > mn2:

                    removed.append(iIndx)
                    _x[iIndx,:] = np.nan
                    _x[:,iIndx] = np.nan
                else:

                    removed.append(jIndx)
                    _x[jIndx,:] = np.nan
                    _x[:,jIndx] = np.nan


    return removed




# Luetaan aineisto:
# dataUrl = "https://raw.githubusercontent.com/RaimoHaikari/data-analytiikka/main/applied-predictive-modelling/data/segmentationOriginal.csv"
dataUrl = "applied-predictive-modelling/data/segmentationOriginal.csv"
segmentationOriginal = pd.read_csv(dataUrl)

# Keskitytään tässä harjoituksessa opetusmaterialiin ja tiputetaan ylimääräiset sarakkeet pois
ehto = segmentationOriginal['Case'] == 'Train'
segData = segmentationOriginal[ehto].copy(deep=True)

segData.drop(['Cell', 'Class', 'Case'], axis=1, inplace = True)
matching = [s for s in segData.columns.values if "Status" in s]
segData.drop(matching, axis=1, inplace = True)


corr_mat = segData.corr()
# corr_mat.style.background_gradient(cmap='coolwarm', axis=None)
# toBeRemoved = findCorrelation(corr_mat, cutoff=0.75)
# print(toBeRemoved)

#foo = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(3,3)
# print(foo)

# print(np.any(foo > 9.1))

# print((foo > 9.1))

np.random.seed([3,1415])
df = pd.DataFrame(
    np.random.randint(10, size=(10, 10)),
    columns=list('ABCDEFGHIJ'))

corr = df.corr()
hc = findCorrelation(corr, cutoff=0.5)
# print(hc)
# print(corr.columns[hc])

acorr = corr.abs()
avg = acorr.mean()