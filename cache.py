import numpy as np

# numpy save float32 raw array
def saveraw(name,data):
  arr = np.array(data,"float32")
  file = open(name,"wb")
  arr.tofile(file)
  file.close()

# import struct
# s = struct.pack('f'*len(floats), *floats)
# f = open('file','wb')
# f.write(s)
# f.close()

# numpy load float32 raw array
def loadraw(name):
  data = np.fromfile(name,dtype="float32")
  return data

# load batch of raw files into vectors, size is size of every vector
def loadraws(paths,sx):
  return np.array([loadraw(re.sub("\.[a-z]*$",f".{sx}",p)) for p in paths])

# load batch of raw files into vectors, size is size of every vector
def saveraws(paths,sx,vectors):
  i = 0
  for path in paths:
    saveraw(re.sub("\.[a-z]*$",f".{sx}",path),vectors[i])
    i += 1

# ------------------------------------------------------------------------------------

# MSG2 print of cached files
def MSG2cached(cache):
  if len(cache.paths2)>0: MSG2(f"{len(cache.paths2)} reduced,")
  if len(cache.paths1)>0:
    MSG2(f"{len(cache.paths1)} percept. vectors")
    if len(cache.paths1all)>len(cache.paths1): MSG2(f"(from {len(cache.paths1all)})")
  elif len(cache.paths1all)>len(cache.paths1): MSG2(f"{len(cache.paths1all)} available percept. vectors")
  MSG3(f"and {len(cache.paths0)} pictures not cached")

# scan cached files
def caching_init(paths,sx1,sx2):
  cache = types.SimpleNamespace()
  cache.sx1	  = sx1	# 
  cache.sx2	  = sx2	# 
  cache.paths0    = []	# paths to pictures (when no cache is available)
  cache.paths1    = []	# paths to model cache
  cache.paths2    = []	# paths to model + reducer
  cache.paths1all = []	# all available model-cached data
  cache.paths0all = []	# when no model cache is available
  MSG1("cache status")

  for p0 in paths:
    p1 = re.sub("\.[a-z]*$",f".{sx1}",p0)
    p2 = re.sub("\.[a-z]*$",f".{sx2}",p0)
    if   os.path.exists(p2):  cache.paths2.append(p0)
    elif os.path.exists(p1):  cache.paths1.append(p0)
    else:		      cache.paths0.append(p0)
    if os.path.exists(p1): cache.paths1all.append(p0)
    else:		   cache.paths0all.append(p0)

  MSG2cached(cache)
  return cache

# ------------------------------------------------------------------------------------
