### NAME
imclust - cluster images

### USAGE
        imclust [OPTIONS] PATH...

### DESCRIPTION
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

Instead of perception/reduction vectors, precomputed vectors can
be used for clustering using the -vec switch.

### CACHING
        Caching of perception and reduction outputs is enabled by default.
        If in current directory or in any parent directory a "CACHE" directory
        is find, it will be used.  Otherwise inputs directory will be used
        for cache files.  Explicit cache directory can by requested by -cd,
        or no caching by -nc.

### OPTIONS
          -h  This help.
          -v  Verbose.
          -f  Force recomputing all data, avoid cached.
       -html  Write html output instead of CSV.
     -o PATH  The base of the output file name.
      -j NUM  Number of threads for loading, dflt. 8.
     -cd DIR  Cache directory to use.
         -nc  Don't cache computed perception nor reduction vectors.
      -c NUM  Requested number of clusters.
      -n NUM  Number of clustering attempts/restarts.
      -m NUM  Limit the max number of images to cluster.
     -mt NUM  Number of members threshold for the cluster to be accepted.
     -dt NUM  Absolute distance threshold from the center cluster, for
              the image to be accepted.
    -pt PERC  Percentual threshold.
      -b NUM  Batch size.
      -r NUM  Reduce vector dimensionality to NUM, dflt. auto from 3072.
     -rp NUM  No. of patterns to train reduction, dflt. auto from 8192.
     -nn STR  Model name, dflt. densenet201 (none, resnet50, resnet152v2, vgg16, inceptionv3, efficientnetb6, densenet121, densenet169, densenet201).
     -cl STR  Clustering algorithm, dflt. km (from km, bkm, kmd).
     -rd STR  Dimensionality reduction, dflt. pca (from none, pca).
      -s STR  Sorting of cluster centers, dflt. tsp (from none, size, tsp).
    -vec STR  Suffix of files with precomputed vectors for every picture,
              for "dir/f_12.jpg" we expect "dir/f_12.vgg" if STR is "vgg".
              Comma separated list of suffixes is allowed, to concatenate
              several vectors into single input for clustering.
         -nm  No metric.
        -jpg  Jpg input files only.

### CLUSTERING
          km  scikit KMeans
         bkm  scikit MiniBatchKMeans
         kmd  scikit KMedoids

### EXAMPLES
        Cluster directory dir according to both .model1 and .model2 precomputed
        raw vectors into 96 clusters as a best from 16 attempts:
        imclust -n 16 -c 96 -vec model1,model2 -o dir-cl.csv dir

### VERSION
imclust 0.5 (c) R.Jaksa 2021

