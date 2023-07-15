from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy
import sys

sheetX_PairD = pd.read_excel("Octahedral_Cage_Dataset.xlsx", sheet_name="KDE on FG_pair_distance")
sheetX_Density = pd.read_excel("Octahedral_Cage_Dataset.xlsx", sheet_name="Structural Features")
sheetY = pd.read_excel("Octahedral_Cage_Dataset.xlsx", sheet_name="Propanolol B.E.");

X_PairD = [list(sheetX_PairD.T[i])[1:69] for i in range(1,3224) if 0<list(sheetY["STDEV"])[i]<8.0]
X_Density = [list(sheetX_Density.T[i])[1:11] for i in range(1,3224) if 0<list(sheetY["STDEV"])[i]<8.0]
Y = [list(sheetY["Average BE"])[i] for i in range(1,3224) if 0<list(sheetY["STDEV"])[i]<8.0]

ndata = len(Y)

X0 = [X_Density[ix]+X_PairD[ix] for ix in range(ndata) if ix%5<4]
Y0 = [Y[ix] for ix in range(ndata) if ix%5<4]
clf = MLPRegressor(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(6), random_state=1, max_iter=20000)
clf.fit(X0, Y0)

Xtest = [X_Density[ix]+X_PairD[ix] for ix in range(ndata) if ix%5>=4 ]
Ytest = [Y[ix] for ix in range(ndata) if ix%5>=4 ]
Y_ML = clf.predict(Xtest)

print("R^2", clf.score(X0, Y0))

ML = pd.DataFrame({'DFTB':Ytest,'Machine Learning':Y_ML})
ML.to_excel("Machine_Learning_Results.xlsx")
