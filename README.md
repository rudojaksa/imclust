### NAME
imclust - cluster images

### USAGE
        imclust [OPTIONS] DIRECTORY...

### DESCRIPTION
Imclust does cluster images in the directory, and produces
a web visualization (or CSV-file output). Clustering is done
in six steps:

   1. loading and resize of images,
   2. perception - transformation of images into vectors,
   3. reduction of dimensionality of vectors (optional),
   4. clustering,
   5. ordering/sorting of clusters,
   6. assembling the visualization or the output data-file.

Caching of perception and reduction outputs can be enabled.

### OPTIONS
          -h  This help.
          -v  Verbose.
          -f  Force recomputing all data, avoid cached.
        -csv  Write csv output instead of html.
     -o PATH  Output file name.
      -j NUM  Number of threads for loading, dflt. 8.
      -cache  Cache computed vectors for every picture (see -vec).
      -c NUM  Requested number of clusters.
      -m NUM  Limit the max number of images to cluster.
     -mt NUM  Number of members threshold for the cluster to be accepted.
     -dt NUM  Absolute distance threshold from the center cluster, for
              the image to be accepted.
    -pt PERC  Percentual threshold.
      -b NUM  Batch size.
      -r NUM  Reduce vector dimensionality to NUM, dflt. auto from 3072.
     -rp NUM  No. of patterns to train reduction, dflt. auto from 8192.
     -nn STR  Model name, dflt. resnet50 (from none, resnet50).
     -cl STR  Clustering algorithm, dflt. km (from km, bkm).
     -rd STR  Dimensionality reduction, dflt. pca (from none, pca).
      -s STR  Sorting of cluster centers, dflt. tsp (from none, size, tsp).
    -vec STR  Suffix of files with precomputed vectors for every picture,
              for "dir/f_12.jpg" we expect "dir/f_12.vgg" if STR is "vgg".

### CLUSTERING
          km  scikit KMeans
         bkm  scikit MiniBatchKMeans

### VERSION
imclust 0.2 (c) R.Jaksa 2021

