from multiprocessing.pool import ThreadPool
from imageio import imread
from skimage.transform import resize

# function to load and resize images
def lrworker(path,size):
  # print(f"---> {path}")
  # image = cv2.imread(path)
  # image = cv2.resize(image,size)
  # cv2.imwrite("/tmp/cv2.png",image)
  try:
    image = imread(str(path))
  except:
    print(f"\nmalformed image: {path}\n")
    return
  image = resize(image,size,anti_aliasing=True)
  return image

# load and resize images in a thread pool
def loadresize(paths,size):
  images = np.empty([0,size[0],size[1],size[2]],dtype=np.float32)
  pool = ThreadPool(THREADS)
  results = []
  for path in paths:
    results.append(pool.apply_async(lrworker,args=(path,size)))
  pool.close()
  pool.join()
  images = np.array([r.get() for r in results]) # assemble the batch-array
  return images

