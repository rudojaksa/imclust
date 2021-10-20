#!/usr/bin/env -S python3 -u # -*- python -*-
import os,sys,time,re,types

# import a silent numpy
import numpy as np
np.warnings.filterwarnings("ignore",category=np.VisibleDeprecationWarning)

# available vs. default perception models (to transform input picture into vector of numbers)
MODELS  = ("none","resnet50","resnet152v2","vgg16","inceptionv3","efficientnetb6")
MODELS += ("densenet121","densenet169","densenet201")
MODEL   = "densenet201"

# available vs. default dimensionality reduction methods
REDIMS = ("none","pca") # TODO: autoencoders
REDIM = "pca"

# available vs. default clustering methods
CLUSTS = ("km","bkm","kmd")
CLUST = "km"
CLUSTERS = 128 # default no. of clusters
CMULT = 1.1 # min ratio of images to clusters (2 = in average 2 images per cluster)

# available vs. default cluster-centers sorting methods
SORTS = ("none","size","tsp")
SORT = "tsp"

# TODO: available vs. default intra-cluster sorting methods
CSORTS = ("none","dist","tsp")
CSORT = "tsp"

# TODO: available vs. default clustering evaluation methods
EVALS = ("none","tsp")
EVAL = "none"

# TODO: available vs. default clusters evaluation methods
CEVALS = ("none","tsp")
CEVAL = "none"

# batchsize and dimensionality reduction size
BATCHSIZE = 2048
REDIMSIZE = 3072 # target size of vector after reduction
REDIMPATS = 8192 # number of patterns to train "reductor"

# no. of loading threads
THREADS = 8

# ------------------------------------------------------------------------------------

def vtime(): return time.clock_gettime(time.CLOCK_MONOTONIC)

def MSG1(str): print(f"{str:>18s}:",end=" ",file=sys.stderr)	# start
def MSG2(str): print(str,end=" ",file=sys.stderr)		# continue
def MSGC(str): print(str,end="",file=sys.stderr)		# progressbar character
def MSG3(str): print(str,file=sys.stderr)			# end
def MSG(hdr,str): MSG1(hdr); MSG3(str)				# hdr+string util function
def MSGE(str): MSG1("error"); MSG3(str); exit()			# error util function

def MSGP(j): # progress-overwrite function, j is progress index
  ch = "."
  if (j+1)% 5==0: ch = ","
  if (j+1)%10==0: ch = ":"
  MSGC(f"\b{ch}")

# return metric number
def metric(num):
  ret = None
  if num > 107374182400: ret = f"{num/1073741824:.0f}G"
  elif num > 1073741824: ret = f"{num/1073741824:.1f}G"
  elif  num > 104857600: ret = f"{num/1048576:.0f}M"
  elif    num > 1048576: ret = f"{num/1048576:.1f}M"
  elif     num > 102400: ret = f"{num/1024:.0f}k"
  elif       num > 1024: ret = f"{num/1024:.1f}k"
  elif        num > 100: ret = f"{num:.0f}"
  else:		         ret = f"{num:.1f}"
  ret = re.sub("\.0([GMk])?$",r"\1",ret)
  return ret

# return time interval
def minsec(sec):
  ret = None
  if sec > 360000: ret = f"{sec/3600:.0f}hr"
  elif sec > 3600: ret = f"{sec/3600:.1f}hr"
  elif   sec > 60: ret = f"{sec/60:.1f}min"
  else:		   ret = f"{sec:.1f}sec"
  ret = re.sub("\.0((hr)|(min)|(sec))?$",r"\1",ret)
  return ret

# return layer size string: WIDTHxHEIGHT*CHANNELS
def lsz(size): return f"{size[0]}x{size[1]}*{size[2]}"
  
# ----------------------------------------------- get directory name from command-line
# include "VERSION.py"

HELP = f"""
NAME
    imclust - cluster images

USAGE
    imclust [OPTIONS] PATH...

DESCRIPTION
    Imclust does cluster images from the directory, and produces
    a CSV-file output (or web visualization).  It accepts also multiple
    directories as input, direct image files, or CSV files where the
    first column are paths to images, or combination of them.

    Clustering is done in six steps:

       1. loading and resize of images,
       2. perception - transformation of images into vectors,
       3. reduction of dimensionality of vectors (optional),
       4. clustering,
       5. ordering/sorting of clusters,
       6. assembling the visualization or the output data-file.

    Caching of perception and reduction outputs can be enabled.

OPTIONS
      -h  This help.
      -v  Verbose.
      -f  Force recomputing all data, avoid cached.
    -csv  Write csv output instead of html.
 -o PATH  The base of the output file name.
  -j NUM  Number of threads for loading, dflt. {THREADS}.
     -nc  Don't cache computed vectors for every picture (also see -vec).
 -cd DIR  Cache dir, implies -cache, if not specified inputs dir is used.
  -c NUM  Requested number of clusters.
  -n NUM  Number of clustering attempts/restarts.
  -m NUM  Limit the max number of images to cluster.
 -mt NUM  Number of members threshold for the cluster to be accepted.
 -dt NUM  Absolute distance threshold from the center cluster, for
          the image to be accepted.
-pt PERC  Percentual threshold.
  -b NUM  Batch size.
  -r NUM  Reduce vector dimensionality to NUM, dflt. auto from {REDIMSIZE}.
 -rp NUM  No. of patterns to train reduction, dflt. auto from {REDIMPATS}.
 -nn STR  Model name, dflt. {MODEL} ({", ".join(MODELS)}).
 -cl STR  Clustering algorithm, dflt. {CLUST} (from {", ".join(CLUSTS)}).
 -rd STR  Dimensionality reduction, dflt. {REDIM} (from {", ".join(REDIMS)}).
  -s STR  Sorting of cluster centers, dflt. {SORT} (from {", ".join(SORTS)}).
-vec STR  Suffix of files with precomputed vectors for every picture,
          for "dir/f_12.jpg" we expect "dir/f_12.vgg" if STR is "vgg".
          Comma separated list of suffixes is allowed, to concatenate
          several vectors into single input for clustering.
     -nm  No metric.
    -jpg  Jpg input files only.

CLUSTERING
      km  scikit KMeans
     bkm  scikit MiniBatchKMeans
     kmd  scikit KMedoids

EXAMPLES
    Cluster directory dir according to both .align and .blend precomputed
    raw vectors into 96 clusters as a best from 16 attempts, result write
    into CSV file:
    imclust -n 16 -c 96 -vec align,blend -csv -o dir-cl.csv dir

VERSION
    imclust {VERSION} (c) R.Jaksa 2021
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
parser.add_argument("-o","--output",type=str)
parser.add_argument("-j","--threads",type=int)
parser.add_argument("-n","--attempts",type=int)

parser.add_argument("-b","--batchsize",type=int)
parser.add_argument("-r","--redimsize",type=int)
parser.add_argument("-rp","--redimpats",type=int)

parser.add_argument("-nn","--nn",type=str,default=MODEL)
parser.add_argument("-cl","--clust",type=str,default=CLUST)
parser.add_argument("-s","--sort",type=str,default=SORT)
parser.add_argument("-rd","--redim",type=str)

parser.add_argument("-vec","--vectors",type=str)
parser.add_argument("-nc","--nocache",action="store_true")
parser.add_argument("-cd","--cachedir",type=str,default="")
parser.add_argument("-nm","--nometric",action="store_true")
parser.add_argument("-jpg","--jpgonly",action="store_true")

parser.add_argument("paths",type=str,nargs='*')
args = parser.parse_args()

if args.help:
    print(HELP)
    exit(0)

VERBOSE = 1 if args.verbose else 0
if args.threads: THREADS = args.threads

args.cache = 1
if args.nocache == 1: args.cache = 0

CACHEDIR = args.cachedir
if args.cachedir != "": args.cache = 1

if not args.nn in MODELS: MSGE(f"unknown perc. model {args.nn}")
else: MODEL = args.nn

if not args.clust in CLUSTS: MSGE(f"unknown clustering {args.clust}")
else: CLUST = args.clust

if not args.redim and args.vectors: REDIM = "none"
if args.redim == None: pass
elif not args.redim in REDIMS: MSGE(f"unknown dim. reduction {args.redim}")
else: REDIM = args.redim

if not args.sort in SORTS: MSGE(f"unknown sorting {args.sort}")
else: SORT = args.sort

# -------------------------------------------------------------------------- filenames

if args.verbose:
  MSG("reqeusted paths",f"{', '.join(args.paths)}")

# output filename
def outputname():
  if args.output: output = args.output
  else:		  output = f"{args.paths[0]}"

  if CLUST != "none":   output += f".{CLUST}{CLUSTERS}"
  if args.vectors:      output += f".{re.sub(',','-',args.vectors)}"
  elif MODEL != "none": output += f".{MODEL}"
  if REDIM != "none":   output += f"-{REDIM}{REDIMSIZE}"
# if SORT  != "none":   output += f".{SORT}"

  output = re.sub("\.csv$","",output)
  output = re.sub("\.html?$","",output)
  return output

if args.clusters: CLUSTERS = args.clusters
MSG("expected name",f"{outputname()}")

# ------------------------------------------------------------------ get list of files
from glob import glob
import random

MSG1("scan paths")
paths = [] # paths to pictures
poor = [] # wrong/missing paths
for name in args.paths:

  # 1st: from CSV (use the 1st column and expect filenames)
  if re.search("\.csv$",name):
    for file in os.popen(f"cut -d ' ' -f 1 {name}").read().rstrip().split("\n"):
      if file[0] == "#": continue
      if os.path.exists(file): paths.append(file)
      else: poor.append(file)

  # 2nd: explicit jpg filenames
  elif re.search("\.jpg$",name):
    if os.path.exists(name): paths.append(name)
    else: poor.append(name)

  # 3rd: explicit png filenames
  elif not args.jpgonly and re.search("\.png$",name):
    if os.path.exists(name): paths.append(name)
    else: poor.append(name)

  # 4th: recursive directory search
  elif os.path.isdir(name):
    paths += glob(name+"/**/*.jpg",recursive=True)
    if not args.jpgonly:
      paths += glob(name+"/**/*.png",recursive=True)

  else: poor.append(name)

MSG2(f"{len(paths)} pictures")

# limit the number of files to process
if args.maximum and args.maximum < len(paths):
  paths = paths[:args.maximum]
  MSG2(f"(limit to {len(paths)})")
MSG3("")

# errors
if poor:
  MSGE(f"wrong pathnames: {', '.join(poor)}")
if len(paths)<1:
  MSGE("cannot proceed without files")

# remove duplicates and shuffle
paths = list(set(paths))
random.shuffle(paths)

#for p in paths: print(p)
# --------------------------------------------------------------------- batching setup
# include "perception.py"

isize,osize,vsize = modelsize(MODEL)

if args.batchsize: BATCHSIZE = args.batchsize
if args.redimsize: REDIMSIZE = args.redimsize
if args.redimpats: REDIMPATS = args.redimpats

if BATCHSIZE > len(paths): BATCHSIZE = len(paths)
if REDIMPATS > len(paths): REDIMPATS = len(paths)
if REDIMSIZE > REDIMPATS: REDIMSIZE = REDIMPATS

MSG("batch size",f"{BATCHSIZE} (aprox. {len(paths)/BATCHSIZE:.0f} batches)")

# -------------------------------------- cached loading of images till reduced vectors
# include "cache.py"
# include "reduction.py"
# include "loading.py"

if args.vectors:
  cache,vsize = caching_init_prec(paths,args.vectors,CACHEDIR,REDIM,REDIMSIZE)
  MODEL = "none"
else:
  cache = caching_init(paths,MODEL,CACHEDIR,REDIM,REDIMSIZE)

REDIMSIZE = min(REDIMSIZE,vsize) # further reduce REDIMSIZE if vector size is too small
  
prcpt = perception_init(cache,MODEL,isize,osize,vsize)
redim =  reduction_init(cache,prcpt,REDIM,REDIMSIZE,REDIMPATS)
vectors =     data_load(cache,prcpt,redim)

# reorder paths, to have the "paths" in the same as "vectors" from loading
paths = []
for path in cache.paths2: paths.append(path)
for path in cache.paths1: paths.append(path)
for path in cache.paths0: paths.append(path)

IMAGES = len(vectors)

# ----------------------------------------------------------------------- cluster them
MSG1("clustering")

maxcl = int(len(paths)/CMULT)				# max. allowed no. of clusters
if len(paths)<2048: CLUSTERS = int(len(paths)/16)	# default for small no. of files
if args.clusters:   CLUSTERS = args.clusters		# requested no. of clusters
if CLUSTERS>maxcl:  CLUSTERS = maxcl			# limited by CMULT
if CLUSTERS<2:	    CLUSTERS = 2			# at least two

T0 = vtime()

# K-means using squared distances
if CLUST == "km":
  from sklearn import cluster
  n_init = 10
  n_init = 2 if IMAGES>8000 else n_init
  n_init = args.attempts if args.attempts else n_init
  clust = cluster.KMeans(n_clusters=CLUSTERS,verbose=VERBOSE,n_init=n_init)
  MSG2(f"scikit KMeans {CLUSTERS} clusters in {n_init} attempts")

if CLUST == "bkm":
  from sklearn import cluster
  n_init = 2
  n_init = 10 if CLUSTERS<2001 else n_init
  n_init = 100 if CLUSTERS<101 else n_init
  n_init = args.attempts if args.attempts else n_init
  clust = cluster.MiniBatchKMeans(n_clusters=CLUSTERS,verbose=VERBOSE,n_init=n_init)
  MSG2(f"scikit MiniBatchKMeans {CLUSTERS} clusters in {n_init} attempts")

# K-medoids using absolute distances
if CLUST == "kmd":
  from sklearn_extra import cluster
  clust = cluster.KMedoids(n_clusters=CLUSTERS)
  MSG2(f"scikit KMedoids {CLUSTERS} clusters")

# clustering itself
dists = clust.fit_transform(vectors) # distances to all cluster centers
idx = clust.predict(vectors)	     # indexes of corresponding clusters
inertia = clust.inertia_	     # method-specific distance of samples to centers
MSG3(f"in {minsec(vtime()-T0)} inertia={inertia:.2f}")

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

# ---------------------------------------------------------------------- sort clusters

# cindex = cluster indexes as sorted according to the number of elements
cindex = list(range(CLUSTERS))

# sort clusters according their size
if SORT == "size":
  elements = [len(x) for x in cluster]
  indexes  = list(range(CLUSTERS))
  cindex = [x for _,x in sorted(zip(elements,indexes),reverse=True)]

# sort clusters by tsp
if SORT == "tsp":
  from sklearn.metrics import pairwise_distances
  from python_tsp.distances import euclidean_distance_matrix
  from python_tsp.exact import solve_tsp_dynamic_programming
  from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
  TSP = []
  TSP.append([]) 
  vectors2 = []
  #print(f"vectors:{len(vectors)}, ordered:{len(ordered)} clusters{CLUSTERS}")
  for j in range(CLUSTERS):
    #print(f"{j}-{ordered[j][0]}",end=" ")
    #print(f"{j}",end=" ")
    vectors2.append(vectors[ordered[j][0]])
  #print("")

  MSG1("distance matrix"); T1=vtime()
  mdist = euclidean_distance_matrix(vectors2)
  MSG3(f"{len(vectors2)}x{len(vectors2)} in {minsec(vtime()-T1)}")

  MSG1(f"{SORT}"); T1=vtime()
  permutation,distance = solve_tsp_simulated_annealing(mdist) 
  MSG3(f"{minsec(vtime()-T1)}")

  cindex = permutation

# TODO: listing only if requested
if 0:
  print("  i: ID  N")
  for j in range(CLUSTERS): print(f"{j+1:3}: {cindex[j]:<3} {elements[cindex[j]]:<3}")

# ------------------------------------------------ copy images according their cluster

# import shutil
# for i in range(IMAGES):
#   if not os.path.exists(f"output/cluster{cluster[i]}"): os.makedirs(f"output/cluster{cluster[i]}")
#   print(f"cp {paths[i]} output/cluster{cluster[i]}")
#   shutil.copy2(f"{paths[i]}",f"output/cluster{cluster[i]}")

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
      section[j] += f"{paths[k]} {jj+1} {dist[k]:.1f} {pdist[k]:.1f} {bad}\n"
    else: # html output
      section[j] += addimg(f"{paths[k]}",f"cluster{jj+1}",f"{pdist[k]:.0f}% {dist[k]:.0f}cm",bad)

MSG1("write output")
output = outputname()

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
# include "metric.py"
metric()

# ------------------------------------------------------------------------------------
