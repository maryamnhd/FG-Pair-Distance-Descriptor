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
#ML.to_excel("Machine_Learning_Results.xlsx")

sys.exit()
#-------------------------------------------------------------------------------


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot




matplotlib.pyplot.rcParams['font.size'] = 50
matplotlib.pyplot.rcParams['font.family'] = 'sans-serif'
matplotlib.pyplot.rcParams['font.weight'] = 'semibold'
matplotlib.pyplot.rcParams['axes.titleweight'] = 'semibold'
matplotlib.pyplot.rcParams['axes.labelweight'] = 'semibold'
matplotlib.pyplot.rcParams['axes.linewidth'] = 6
matplotlib.pyplot.rcParams['xtick.major.size'] = 8
matplotlib.pyplot.rcParams['xtick.major.width'] = 3
matplotlib.pyplot.rcParams['xtick.minor.width'] = 2
matplotlib.pyplot.rcParams['ytick.major.size'] = 8
matplotlib.pyplot.rcParams['ytick.major.width'] = 3
matplotlib.pyplot.rcParams['ytick.minor.width'] = 2
matplotlib.pyplot.figure(figsize=(20, 20), dpi=200)


xmin, xmax = max(list(Ytest)+list(Y1))*0.98, min(list(Ytest)+list(Y1))*1.02

matplotlib.pyplot.scatter(Ytest,Y1, s =150)
matplotlib.pyplot.axline((0, 0), slope=1, color="red",linewidth=4)
matplotlib.pyplot.xlabel('DFTB B.E. (kJ/mol)')
matplotlib.pyplot.ylabel('Machine Learning B.E. (kJ/mol)')   
matplotlib.pyplot.xlim(xmin,xmax)
matplotlib.pyplot.ylim(xmin,xmax)

matplotlib.pyplot.savefig("MachineLearning_std8_distribution.png", transparent=True, bbox_inches='tight')
matplotlib.pyplot.close()
