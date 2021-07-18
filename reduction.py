# include "loading.py"
from sklearn.decomposition import PCA

def reduction_init(cache,prcpt,REDIM,REDIMSIZE,REDIMPATS):
  redim = types.SimpleNamespace()
  redim.name = REDIM
  redim.size = REDIMSIZE
  redim.pats = REDIMPATS
  redim.model = None
  cached = [] # newly-cached list for perception vectors

  # redim not needed
  if cache.paths2 and not cache.paths1 and not cache.paths0: return redim
  # none mode
  if redim.name == "none": return redim

  # trainable modes
  MSG(f"{redim.name} setup",f"{prcpt.vsize} -> {redim.size} dimensions (from {redim.pats} samples)")

  # engine init
  redim.model = PCA(n_components=redim.size)
  # redim.model = cuml.PCA(n_components=redim.size)

  # loading
  MSG1(f"{redim.name} train"); T1 = vtime()
  j = 0 # batch index
  all_vectors = np.empty([0,prcpt.vsize],dtype=np.float32)

  # 1st: load cached vectors
  i = 0 # image index
  end = min(len(cache.paths1all),redim.pats)
  while i < end:
    i2 = i + BATCHSIZE
    if i2 > end: i2 = end
    MSGC("c")
    vectors = loadraws(cache.paths1all[i:i2],cache.sx1)
    MSGP(j)
    all_vectors = np.concatenate((all_vectors,vectors),0)
    i += len(vectors)
    j += 1
  
  # 2nd: load raw images
  i = 0
  end = min(len(cache.paths0all),redim.pats)
  while i < end:
    i2 = i + BATCHSIZE
    if i2 > end: i2 = end
    MSGC("l")
    images = loadresize(cache.paths0all[i:i2],prcpt.isize)
    MSGC("\bn")
    vectors = transform(prcpt,images)
    if args.cache:
      saveraws(cache.paths0all[i:i2],cache.sx1,vectors)
      cached += cache.paths0all[i:i2]
    MSGP(j)
    all_vectors = np.concatenate((all_vectors,vectors),0)
    i += len(vectors)
    j += 1

  # 3rd: train redim
  mem = all_vectors.nbytes
  MSGC("R"); T2 = vtime()
  redim.model.fit(all_vectors)

  # end
  MSGP(j); T3 = vtime()
  MSG2(f" {metric(mem)}B of vectors,")
  MSG2(f"{minsec(T2-T1)} loading,")
  MSG2(f"{minsec(T3-T2)} training time")
  MSG3("")

  # move newly cached vectors from paths0 to paths1
  if args.cache and cached:
    cache.paths0 = list(set(cache.paths0).difference(set(cached)))	# paths0 = paths0 - cached
    cache.paths1 = list(set(cache.paths1 + cached))			# paths1 = paths1 + cached
    MSG1("newly cached")
    MSG2(f"{len(cached)} pepcept. vectors, so we got")
    MSG2cached(cache)

  return redim

