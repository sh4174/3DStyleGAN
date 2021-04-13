
import os
import sys
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Create separate train/validation dataset from given folder. Only creates symlinks, does not duplicate data. Only works for Linux/MacOs')
parser.add_argument('-f', '--folder', help='Input folder', dest='folder')

args = parser.parse_args()

fldTrain = args.folder + '_train'
fldTest = args.folder + '_test'

os.system('mkdir -p ' + fldTrain)
os.system('mkdir -p ' + fldTest)


files = [f for f in listdir(args.folder) if isfile(join(args.folder, f))]
print(files[:10])
nrFiles = len(files)

testPerc = 0.1 # put 10% of images in test set
nrTest = int(nrFiles * testPerc)

indTest = np.random.choice(nrFiles, nrTest, replace='False')

for i, f in enumerate(files):
  print('%d/%d' % (i, nrFiles))
  if i in indTest:
    os.system('ln -s %s/%s %s/%s' % (args.folder, f, fldTest, f))
  else:
    os.system('ln -s %s/%s %s/%s' % (args.folder, f, fldTrain, f))

