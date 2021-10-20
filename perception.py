
def  modelname(model): return model._name
def  inputsize(model): return model._feed_input_shapes[0][1:]
def outputsize(model): return model.outputs[0].shape[1:]

# ------------------------------------------------------------------------------------

# just return the input/output/vector sizes, without model loading
def modelsize(name):
  if name == "none":		return (128,128,3),(128,128,3),49152
  if name == "resnet50":	return (224,224,3),(7,7,2048),100352
  if name == "resnet152v2":	return (224,224,3),(7,7,2048),100352
  if name == "vgg16":		return (224,224,3),(7,7,512) ,25088
  if name == "inceptionv3":	return (224,224,3),(5,5,2048),51200
  if name == "efficientnetb6":	return (224,224,3),(7,7,2304),112896
  if name == "densenet121":	return (224,224,3),(7,7,1024),50176
  if name == "densenet169":	return (224,224,3),(7,7,1664),81536
  if name == "densenet201":	return (224,224,3),(7,7,1920),94080

# get sizes from model...
def realsize(prcpt):
  if prcpt.name == "none": return modelsize(prcpt.name)
  osize = outputsize(prcpt.model)
  return inputsize(prcpt.model),osize,osize[0]*osize[1]*osize[2]
  
# load model by name and return "prcpt" structure
def perception_init(cache,name,isize,osize,vsize):
  prcpt = types.SimpleNamespace()
  prcpt.name = name

  if name == "none":
    prcpt.vsize = vsize
  else:
    prcpt.isize,prcpt.osize,prcpt.vsize = modelsize(name)

  if name == "none": return prcpt	# no model for none model
  if cache.paths0: pass			# needs model for loading
  elif cache.paths0all: pass		# probably needs for redim
  else: return prcpt

  # tensorflow models start here
  MSG1("init perception")
  if not VERBOSE: os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  import tensorflow as tf

  if name == "resnet50":
    prcpt.model = tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet",input_shape=(224,224,3))
  if name == "resnet152v2":
    prcpt.model = tf.keras.applications.ResNet152V2(include_top=False,weights="imagenet",input_shape=(224,224,3))
  if name == "vgg16":
    prcpt.model = tf.keras.applications.VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))
  if name == "inceptionv3":
    prcpt.model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
  if name == "efficientnetb6":
    prcpt.model = tf.keras.applications.efficientnet.EfficientNetB6(include_top=False,weights="imagenet",input_shape=(224,224,3))
  if name == "densenet121":
    prcpt.model = tf.keras.applications.densenet.DenseNet121(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
  if name == "densenet169":
    prcpt.model = tf.keras.applications.densenet.DenseNet169(include_top=False,weights="imagenet",input_shape=(224,224,3))
  if name == "densenet201":
    prcpt.model = tf.keras.applications.densenet.DenseNet201(include_top=False,weights="imagenet",input_shape=(224,224,3)) 
    
  prcpt.name = modelname(prcpt.model)
  prcpt.isize,prcpt.osize,prcpt.vsize = realsize(prcpt)
  MSG3(f"{prcpt.name} {lsz(prcpt.isize)} -> {lsz(prcpt.osize)} = {prcpt.vsize}")
  if isize!=prcpt.isize or osize!=prcpt.osize or vsize!=prcpt.vsize:
    MSGE(f"perception size mismatch, was: {lsz(isize)} -> {lsz(osize)} = {vsize}")
  return prcpt

# ------------------------------------------------------------------------------------

def perceive(prcpt,images):
  vectors = prcpt.model.predict(images)
  vectors = vectors.reshape(images.shape[0],-1)
  return vectors

# ------------------------------------------------------------------------------------
