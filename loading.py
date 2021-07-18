# include "loadresize.py"

# the input loop: load + resize + nn + redim
def data_load(cache,prcpt,redim):

  # data loading setup
  needmodel = 0
  MSG1("loading setup")
  if cache.paths2: MSG2(f"{len(cache.paths2)} reduced: just load,")
  if cache.paths1: MSG2(f"{len(cache.paths1)} percept.: load -> {redim.name},")
  if cache.paths0:
    MSG2(f"{len(cache.paths0)} pictures: load+resize -> {prcpt.name} -> {redim.name},")
    needmodel = 1
  MSG3("\b\b ")

  # loading itself
  MSG1("load"); T1 = vtime()
  all_vectors = np.empty([0,redim.size],dtype=np.float32)
  images  = []	# 
  cached1 = []	# newly-cached list for perception vectors
  cached2 = []	# newly-cached list for dim-reduced vectors
  ibytes = 0	# accumulated hypothetical space needed for images
  j = 0		# batch index

  # 1st: cached already reduced dimensions -> just load
  i = 0 # image index
  end = len(cache.paths2)
  while i < end:
    i2 = i + BATCHSIZE
    if i2 > end: i2 = end
    MSGC("C")
    vectors = loadraws(cache.paths2[i:i2],cache.sx2)
    MSGP(j)
    all_vectors = np.concatenate((all_vectors,vectors),0)
    i += len(vectors)
    j += 1

  # 2nd: cached nn vectors -> only reduce dimensionality
  i = 0
  end = len(cache.paths1)
  while i < end:
    i2 = i + BATCHSIZE
    if i2 > end: i2 = end
    MSGC("c")
    vectors = loadraws(cache.paths1[i:i2],cache.sx1)
    MSGC("\br")
    vectors = redim.model.transform(vectors)
    if args.cache:
      saveraws(cache.paths1[i:i2],cache.sx2,vectors)
      cached2 += cache.paths1[i:i2]
    MSGP(j)
    all_vectors = np.concatenate((all_vectors,vectors),0)
    i += len(vectors)
    j += 1

  # 3rd: raw images -> load, resize, nn infer, reduce dim.
  i = 0
  end = len(cache.paths0)
  while i < end:
    i2 = i + BATCHSIZE
    if i2 > end: i2 = end
    MSGC("l")
    images = loadresize(cache.paths0[i:i2],prcpt.isize)
    ibytes += images.nbytes
    MSGC("\bn")
    vectors = transform(prcpt,images)
    if args.cache:
      saveraws(cache.paths0[i:i2],cache.sx1,vectors)
      cached1 += cache.paths0[i:i2]
    MSGC("\br")
    vectors = redim.model.transform(vectors)
    if args.cache:
      saveraws(cache.paths0[i:i2],cache.sx2,vectors)
      cached2 += cache.paths0[i:i2]
    MSGP(j)
    all_vectors = np.concatenate((all_vectors,vectors),0)
    i += len(vectors)
    j += 1

  # end
  T2 = vtime()

  MSG2("")
  if len(images): MSG2(f"{metric(ibytes)}B of {len(images)} images,")
  MSG2(f"{metric(all_vectors.nbytes)}B of {len(all_vectors)} vectors")
  MSG3(f"in {j} batches in {minsec(T2-T1)}")

  if cached1 or cached2:  MSG1("newly cached")
  if cached1:		  MSG2(f"{len(cached1)} percept. vectors")
  if cached1 and cached2: MSG2("and")
  if cached2:		  MSG2(f"{len(cached2)} reduced vectors")
  if cached1 or cached2:  MSG3("")

  return all_vectors

