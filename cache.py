import numpy as np

# numpy save float32 raw array
def saveraw(name,data):
  arr = np.array(data,"float32")
  file = open(name,"wb")
  arr.tofile(file)
  file.close()

# numpy load float32 raw array, as a concatenation of vectors from multiple files
def loadraw(base,sx):
  return np.concatenate([np.fromfile(re.sub("\.[a-z]*$",f".{s}",base),dtype="float32") for s in sx.split(",")])

# load batch of raw files into vectors, size is size of every vector
def loadraws(paths,sx):
  return np.array([loadraw(p,sx) for p in paths])

# load batch of raw files into vectors, size is size of every vector
def saveraws(paths,sx,vectors):
  i = 0
  for path in paths:
    saveraw(re.sub("\.[a-z]*$",f".{sx}",path),vectors[i])
    i += 1

# ------------------------------------------------------------------------------------

# MSG2 print of cached files
def MSG2cached(cache):
  l1  = len(cache.paths1)
  l2  = len(cache.paths2)
  l1a = len(cache.paths1all)
  if l2: MSG2(f"{l2} reduced,")
  if l1:
    MSG2(f"{l1} percepts")
    if l1a>l1: MSG2(f"(from {l1a})")
  elif l1a>l1: MSG2(f"{l1a} available percepts")
  if l1 or l2 or l1a>l1: MSG2("and")
  MSG3(f"{len(cache.paths0)} pictures not cached")

# scan cached files
def caching_init(paths,model,redim,redimsize):
  if redim == "none": redimsize = ""
  cache = types.SimpleNamespace()
  cache.sx1	  = f"{model}" # actual suffixe for perception cached files
  cache.sx2	  = f"{model}-{redim}{redimsize}" # perception+redim
  cache.sx1s	  = cache.sx1
  cache.paths0    = []	# paths to pictures (when no cache is available)
  cache.paths1    = []	# paths to model cache
  cache.paths2    = []	# paths to model + reducer
  cache.paths1all = []	# all available model-cached data
  cache.paths0all = []	# when no model cache is available

  MSG1("cache files")
  if redim == "none": MSG3(f"{cache.sx1} cache suffixes")
  else: MSG3(f"{cache.sx1} and {cache.sx2} cache suffixes")

  MSG1("cache status")

  if redim == "none":
    for p0 in paths:
      p1 = re.sub("\.[a-z]*$",f".{cache.sx1}",p0)
      if os.path.exists(p1):
        cache.paths1.append(p0)
        cache.paths1all.append(p0)
      else:
        cache.paths0.append(p0)
        cache.paths0all.append(p0)
  else:
    for p0 in paths:
      p1 = re.sub("\.[a-z]*$",f".{cache.sx1}",p0)
      p2 = re.sub("\.[a-z]*$",f".{cache.sx2}",p0)
      if   os.path.exists(p2):	cache.paths2.append(p0)
      elif os.path.exists(p1):	cache.paths1.append(p0)
      else:			cache.paths0.append(p0)
      if os.path.exists(p1):	cache.paths1all.append(p0)
      else:			cache.paths0all.append(p0)

  MSG2cached(cache)
  return cache

# input vectors from precomputed files by suffix
def caching_prec(paths,sx1s,redim,redimsize):
  if redim == "none": redimsize = ""
  cache = types.SimpleNamespace()
  sx1a = sx1s.split(",")			# array of (input) suffixes
  cache.sx1s = sx1s				# save orig string for loader
  cache.sx1 = re.sub(",","",sx1s)		# actual (output) file suffix
  cache.sx2 = f"{cache.sx1}-{redim}{redimsize}"	# inputs+redim
  cache.paths0    = []	# paths to pictures (when no cache is available)
  cache.paths1    = []	# paths to model cache
  cache.paths2    = []	# paths to model + reducer
  cache.paths1all = []	# all available model-cached data
  cache.paths0all = []	# when no model cache is available
  sizes = []		# list of sizes strings
  size = 0		# accumulated total size

  MSG("cache files",f"{' '.join(sx1a)} cache suffixes")
  MSG1("cache status")
  
  for p0 in paths:

    # for EVERY suffix, the file must exist
    allok = 1
    for sx in sx1a:
      p = re.sub("\.[a-z]*$",f".{sx}",p0)
      if not os.path.exists(p): allok = 0
    if not allok: continue

    # remember the image path
    cache.paths1.append(p0)
    cache.paths1all.append(p0)

    # get the size from 1st files
    if not size:
      for sx in sx1a:
        p = re.sub("\.[a-z]*$",f".{sx}",p0)
        sz = int(os.path.getsize(p)/4) # fro float 32
        size += sz
        sizes.append(str(sz))

  # fix the suffix
  if redim != "none" and vsize < redimsize:
    cache.sx2 = f"{cache.sx1}-{redim}{vsize}"

  MSG2cached(cache)
  MSG1("vectors size")
  if len(sizes)>1: MSG2(" + ".join(sizes) + " =")
  MSG3(f"{size:.0f}")
  return cache,size

# ------------------------------------------------------------------------------------
