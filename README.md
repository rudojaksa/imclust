### NAME
imclust - cluster images

### USAGE
        imclust [OPTIONS] DIRECTORY...

### DESCRIPTION
Imclust does cluster images in the directory, and produces
a web visualization.

### OPTIONS
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

### CLUSTERING
          km  scikit KMeans
         bkm  scikit MiniBatchKMeans

### VERSION
imclust 0.1 (c) R.Jaksa 2021

