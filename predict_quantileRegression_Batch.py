import sys
import os
#from import_file import import_file
from quantileRegression import quantileRegression

#QReg = import_file("/mnt/t3nfs01/data01/shome/krgedia/Workspace/lxplus/qreg/quantileRegression")

# Nevt passing selection
# mc   4545316 : train on 2000000
# data 1441902 : train on all

class mycolors:
   red = '\033[91m'
   green = '\033[92m'
   blue = '\033[94m'
   yan = '\033[96m'
   cWhite = '\033[97m'
   yellow = '\033[93m'
   magenta = '\033[95m'
   grey = '\033[90m'
   black = '\033[90m'
   default = '\033[0m'


print mycolors.green+"Training quantile regressions on"+mycolors.default
dataMC = sys.argv[1]
print "Data/MC = ", dataMC
fileDir = sys.argv[2]
treeDir = sys.argv[3]
Y = sys.argv[4]
print "Y = ", Y
#quantiles =  sys.argv[5]
#print "Quantile = ", quantiles


qr = quantileRegression(dataMC,fileDir,treeDir)
qr.loadDFh5_test()
qr.loadWeights(Y)
qr.loadClass(Y)
qr.correctY(Y)

