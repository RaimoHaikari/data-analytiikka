# %%

# https://stackoverflow.com/questions/44889508/remove-highly-correlated-columns-from-a-pandas-dataframe

# 
# Ladataan käytettävät paketit:
import numpy as np
import pandas as pd

from copy import copy, deepcopy

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def findCorrelation(x, cutoff = 0.9, verbose = False):

    removed = []

    # Varmistetaan, että aineiston tietotyypit ovat float tyyppisiä
    if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

    # Irroitetaan arvot Numpy-taulukkoon
    _x = x.values


    # Varmistetaan, että matriisi symmetrinen
    if check_symmetric(_x) == False:
        print("Ei ole symmetrinen")
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


    return list(x.columns[removed])

# Alustetaan aineisto
np.random.seed([3,1415])
df = pd.DataFrame(
    np.random.randint(10, size=(10, 10)),
    columns=list('ABCDEFGHIJ')
)
# print(df)


# Lasketaan korrelaatiomatriisi
corr = df.corr()
# print(corr)


hc = findCorrelation(corr, cutoff=0.5)
print(hc)

# corr.to_csv('out.csv', index=False, sep="\t")