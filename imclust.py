#!/usr/bin/env -S python3 -u # -*- python -*-
import os,sys,time,re

# ------------------------------------------------------------------------------------

def vtime(): return time.clock_gettime(time.CLOCK_MONOTONIC)

def MSG1(str): print(f"{str:>18s}:",end=" ",file=sys.stderr)	# start
def MSG2(str): print(str,end=" ",file=sys.stderr)		# continue
def MSGC(str): print(str,end="",file=sys.stderr)		# progressbar character
def MSG3(str): print(str,file=sys.stderr)			# end
def MSGE(str): MSG1("error"); MSG3(str); exit()

# ----------------------------------------------- get directory name from command-line

HELP = f"""
NAME
    imclust - cluster images

USAGE
    imclust [OPTIONS] DIRECTORY...

DESCRIPTION
    Imclust does cluster images in the directory, and produces
    a web visualization.

OPTIONS
      -h  This help.
      -v  Verbose.
    -csv  Write csv output instead of html.
 -o PATH  Output file name.
  -c NUM  Requested number of clusters.
  -m NUM  Limit the max number of images to cluster.
  -b NUM  Batch size.
 -b1 NUM  1st batch size (for PCA fit).
  -f STR  Clustering function: km,bkm.
 -mt NUM  Number of members threshold for the cluster to be accepted.
 -dt NUM  Absolute distance threshold from the center cluster, for
          the image to be accepted.
-pt PERC  Percentual threshold.

CLUSTERING
      km  scikit KMeans
     bkm  scikit MiniBatchKMeans

VERSION
    imclust 0.1 (c) R.Jaksa 2021
"""

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h","--help",action="store_true")
parser.add_argument("-v","--verbose",action="store_true")
parser.add_argument("-csv","--csv",action="store_true")
parser.add_argument("-c","--clusters",type=int)
parser.add_argument("-m","--maximum",type=int)
parser.add_argument("-dt","--distthr",type=int)
parser.add_argument("-pt","--percthr",type=int)
parser.add_argument("-f","--func",type=str,default="km")
parser.add_argument("-o","--output",type=str)
parser.add_argument("-b1","--batchsize1",type=int)
parser.add_argument("-b","--batchsize",type=int)
parser.add_argument("path",type=str,nargs='*')
args = parser.parse_args()

if args.help:
    print(HELP)
    exit(0)

VERBOSE = 1 if args.verbose else 0

funcs = ("km","bkm")
if not args.func in funcs: MSGE(f"unknown clustering {args.func}")

# ---------------------------------------------------------- get image names from dirs
from glob import glob
import random
MSG1("scan paths")

path = []
for dir in args.path:
  path += glob(dir+"/**/*.png",recursive=True)
  path += glob(dir+"/**/*.jpg",recursive=True)
random.shuffle(path)
MSG2(f"{len(path)} files")

if args.maximum and args.maximum < len(path):
  path = path[:args.maximum]
  MSG2(f"(limit to {len(path)})")

MSG3("")
if len(path)<1: MSGE("cannot proceed without files")

#for p in path: print(p)
# ------------------------------------------------------------------------- load model
if not VERBOSE: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
MSG1("load model")

SIZE = (224,224,3)
model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=SIZE)
MSG3(f"{model._name} (input {model._feed_input_shapes[0][1]}x{model._feed_input_shapes[0][1]})")

SIZE = model._feed_input_shapes[0][1:]
VSIZE = model.outputs[0].shape[1] * model.outputs[0].shape[2] * model.outputs[0].shape[3]

MSG1("resize to")
MSG3(f"to {SIZE[0]}x{SIZE[1]} {VSIZE}")

# ------------------------------------------------------------------------ load images

import numpy as np
np.warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)
  
from imageio import imread
from skimage.transform import resize
from sklearn.decomposition import PCA

BATCHSIZE1 = 4096 # 1st batch
BATCHSIZE  = 1024 # other
if args.batchsize1: BATCHSIZE1 = args.batchsize1
if args.batchsize:  BATCHSIZE  = args.batchsize

if BATCHSIZE1 > len(path): BATCHSIZE1 = len(path)
if BATCHSIZE  > len(path): BATCHSIZE  = len(path)

MSG1("loading setup")
MSG3(f"load-resize-nn-pca {1+(len(path)-BATCHSIZE1)/BATCHSIZE:.0f} batches from {len(path)} images")

# PCA setup
PCASIZE = min(BATCHSIZE1,VSIZE)
pca = PCA(n_components=PCASIZE)
#pca = cuml.PCA(n_components=PCASIZE)
batch1 = 1 # whether it is the 1st batch
MSG1("pca")
MSG3(f"to {PCASIZE} components")

# loading itself
MSG1("load")
i = 0 # image index
j = 0 # batch index
ibytes = 0
rbytes = 0
vbytes = 0
vectors = np.empty([0,PCASIZE],dtype=np.float32)
batchsize = BATCHSIZE1
while i<len(path):
  i2 = i+batchsize
  if i2>len(path): i2 = len(path) 

  # load
  MSGC("l")
  #images = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]],dtype=object)
  images = np.array([imread(str(p)).astype(np.float32) for p in path[i:i2]])
  i += len(images)
  ibytes += images.nbytes

  # resize
  MSGC("\br")
  resized = np.asarray([resize(image,SIZE,0) for image in images])
  rbytes += resized.nbytes

  # nn transform
  MSGC("\bn")
  vector = model.predict(resized)
  vector = vector.reshape(resized.shape[0],-1)
  vbytes += vector.nbytes

  # pca
  if batch1:
    MSGC("\bP")
    pca.fit(vector)
  MSGC("\bp")
  vector = pca.transform(vector)

  # progress
  ch = "."
  if batch1: ch = "o"
  if (j+1)% 5==0: ch = ","
  if (j+1)%10==0: ch = ":"
  MSGC(f"\b{ch}")
  
  # append
#  vectors = tf.concat([vectors,vector],0)
  vectors = np.concatenate((vectors,vector),0)

  # iterate
  if batch1:
    batchsize = BATCHSIZE
    batch1 = 0
  j += 1

MSG3("")

MSG1("memory consumed")
MSG2(f"{ibytes/1073741824:.1f}GB images, ")
MSG2(f"{rbytes/1073741824:.1f}GB resized images, ")
MSG3(f"{vbytes/1073741824:.1f}GB vectors")
#MSG3(f"{len(images)} images ({images.nbytes/1073741824:.1f}GB)")
# MSG3(f"{images.nbytes/1073741824:.1f}GB")
#MSG3(f"got {len(vector)} vectors {vector[0].shape[0]} long")

IMAGES = len(vectors)

# ----------------------------------------------------------------------- cluster them
from sklearn import cluster
MSG1("clustering")

CLUSTERS = 128
if len(path)<2048: CLUSTERS = int(len(path)/16)
if CLUSTERS<2:	   CLUSTERS = 2
if args.clusters:  CLUSTERS = args.clusters

T0 = vtime()

if args.func == "km":
  n_init = 10
  n_init = 2 if IMAGES>8000 else n_init
  knn = cluster.KMeans(n_clusters=CLUSTERS,verbose=VERBOSE,n_init=n_init)
  MSG2(f"scikit KMeans {CLUSTERS} clusters in {n_init} attempts")

if args.func == "bkm":
  n_init = 2
  n_init = 10 if CLUSTERS<2001 else n_init
  n_init = 100 if CLUSTERS<101 else n_init
  knn = cluster.MiniBatchKMeans(n_clusters=CLUSTERS,verbose=VERBOSE,n_init=n_init)
  MSG2(f"scikit MiniBatchKMeans {CLUSTERS} clusters in {n_init} attempts")

dists = knn.fit_transform(vectors) # distances to all cluster centers
idx = knn.predict(vectors)	  # indexes of corresponding clusters
MSG3(f"in {vtime()-T0:.1f}sec")

# distances to the closest cluster
dist = [np.inf] * IMAGES
for i in range(IMAGES):
  dist[i] = dists[i][idx[i]]
  #print(f"{i} {idx[i]} {dist[i]}")

# ------------------------------------------------------------------ organize clusters

# get per-cluster indexes and distances (needed to allow sorting)
cluster = []
cdist = []
for j in range(CLUSTERS):
  cluster.append([])
  cdist.append([])
for i in range(IMAGES):
  j = idx[i]
  cluster[j].append(i)
  cdist[j].append(dist[i])
#for j in range(CLUSTERS): print(f"{j}: {cluster[j]} {cdist[j]}")

# ordered indexes of images in clusters
ordered = [None]*CLUSTERS
for j in range(CLUSTERS):
  ordered[j] = [x for _,x in sorted(zip(cdist[j],cluster[j]))]
# for j in range(CLUSTERS): print(f"{j}: {ordered[j]} {cdist[j]}")

# obtain maximum distance from the cluster center
maxdist = 0
for i in range(IMAGES):
  if dist[i] > maxdist: maxdist = dist[i]

# compute percentage distances from the center for every image
pdist = [None]*IMAGES
for j in range(CLUSTERS):
  if ordered[j] == []: continue	# skip empty clusters
  d0 = dist[ordered[j][0]]		# distance of the closest image to the center
  for i in range(len(ordered[j])):
    k = ordered[j][i]
    pdist[k] = (maxdist-dist[k])/(maxdist-d0)*100

# compute active number of images in cluster (above thresholds)
above = [np.inf]*CLUSTERS
distthr = 0
percthr = 0
for j in range(CLUSTERS):
  if ordered[j] == []: continue
  for i in range(len(ordered[j])):
    k = ordered[j][i]
    if args.distthr and dist[k]>args.distthr:
      if i<above[j]:
        above[j] = i
        distthr += 1
    if args.percthr and pdist[k]<args.percthr:
      if i<above[j]:
        above[j] = i
        percthr += 1

if args.distthr:
  MSG1("distance threshold")
  MSG3(f"{distthr} images above {args.distthr}cm")
if args.percthr:
  MSG1("perc threshold")
  MSG3(f"{percthr} images below {args.percthr}%")

# ------------------------------------------------- sort clusters according their size

# cindex = cluster indexes as sorted according to the number of elements
elements = [len(x) for x in cluster]
indexes  = list(range(CLUSTERS))
cindex = [x for _,x in sorted(zip(elements,indexes),reverse=True)]
# for i in range(CLUSTERS): print(f"{cindex[i]}: {len(ordered[cindex[i]])}")
  
# ------------------------------------------------ copy images according their cluster

# import shutil
# for i in range(IMAGES):
#   if not os.path.exists(f"output/cluster{cluster[i]}"): os.makedirs(f"output/cluster{cluster[i]}")
#   print(f"cp {path[i]} output/cluster{cluster[i]}")
#   shutil.copy2(f"{path[i]}",f"output/cluster{cluster[i]}")

# ------------------------------------------------------------ make html or csv output
# from web import *
# include "web.py"

# distribute all images into per-cluster html sections
section = [""]*CLUSTERS
for jj in range(CLUSTERS):		# j = cluster index as from kmeans
  j = cindex[jj]
  for i in range(len(ordered[j])):	# i = order in_the_cluster
    k = ordered[j][i]			# k = image_index
    bad = 1 if i>=above[j] else 0;
    if args.csv: # csv output
      section[j] += f"{path[k]} {jj+1} {dist[k]:.1f} {pdist[k]:.1f} {bad}\n"
    else: # html output
      section[j] += addimg(f"{path[k]}",f"cluster{jj+1}",f"{pdist[k]:.0f}% {dist[k]:.0f}cm",bad)

# output filename
output = "clust"
if args.output: output = args.output
output = re.sub("\.csv$","",output)
output = re.sub("\.html?$","",output)
MSG1("write output")

# save csv
BODY = ""
if args.csv:
  output = f"{output}.csv"
  for i in range(CLUSTERS):
    BODY += section[cindex[i]]
  with open(output,"w") as fd:
    fd.write("#path cluster dist pdist bad\n")
    fd.write(BODY)

# save html
else:
  output = f"{output}.html"
  for i in range(CLUSTERS):
    BODY += f"<h2>cluster {i+1}</h2>\n"
    BODY += section[cindex[i]]
    BODY += "\n\n"
  html = HTML.format(BODY=BODY,CSS=CSS)
  with open(output,"w") as fd:
    fd.write(html)

MSG3(output)
# ------------------------------------------------------------------------------------
