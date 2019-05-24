import model  as model
import os 
import tensorflow as tf
import json
import numpy as np
import PIL.Image
from multiprocessing import dummy as multiprocessing
import math
from datetime import datetime

def create_session():
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.operation_timeout_in_ms = int(100000000)
    return tf.InteractiveSession(graph=graph, config=config)

def load_image_from_file(filename, shape):
  """Given a filename, try to open the file. If failed, return None.

  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled

  Returns:
    the image if succeeds, None if fails.

  Rasies:
    exception if the image was not the right shape.
  """
  if not tf.gfile.Exists(filename):
    return None
  try:
    # ensure image has no transparency channel
    img = np.array(PIL.Image.open(tf.gfile.Open(filename, 'rb')).convert(
        'RGB').resize(shape, PIL.Image.BILINEAR))
    # Normalize pixel values to between 0 and 1.
    img = np.float32(img) / 255.0
    if not (len(img.shape) == 3 and img.shape[2] == 3):
      return None
    else:
      return img

  except Exception as e:
    print(e)
    return None
  return img

def load_images_from_files(filenames, max_imgs=500,
                           do_shuffle=True,
                           shape=(299, 299)):
  """Return image arrays from filenames.

  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    do_shuffle: before getting max_imgs files, shuffle the names or not
    shape: desired shape of the image

  Returns:
    image arrays

  """
  imgs = []
  # First shuffle a copy of the filenames.
  filenames = filenames[:]
  if do_shuffle:
    np.random.shuffle(filenames)

  pool = multiprocessing.Pool(100)
  imgs = pool.map(
      lambda filename: load_image_from_file(filename, shape),
      filenames[:max_imgs])
  imgs = [img for img in imgs if img is not None]
  if len(imgs) <= 1:
    raise ValueError('You must have more than 1 image in each class to run TCAV.')

  return np.array(imgs)


def main():
    bottleneck =  'mixed4d'   
    max_examples = 100
    sess = create_session()
    wd = '/media/adri/Spare/acts_grads/'
    classes = None
    with open('../../concept-vis/data/classes.json') as f:
        classes = json.loads(f.read())

    images_path = '/media/adri/Spare/tiny-imagenet-200/train/'


    GRAPH_PATH =  '../../trained_models/google_net_inception_v1/tensorflow_inception_graph.pb'  # trained model location 
    LABEL_PATH = '../../trained_models/google_net_inception_v1/imagenet_comp_graph_label_strings.txt' # output labels
        
    mymodel = model.GoolgeNetWrapper_public(sess,
                                            GRAPH_PATH,
                                            LABEL_PATH)

    for i, item in enumerate(classes):
        if not os.path.isfile(wd + item['id'] + "_sorted_grads"):
            start_time = datetime.now()

            imgs = sorted(os.listdir(images_path + item['id'] + '/images'))
            filenames = [images_path + item['id'] + '/images/' + x for x in imgs] 
            examples = load_images_from_files(filenames)
            with open(wd + item['id']+ '_names','w') as f:
                f.write('\n'.join(imgs))
            acts_all = []
            grads_all = []
            for batch in range(math.ceil(len(imgs)/max_examples)):
                try:     
                    acts = mymodel.run_examples(examples[batch*max_examples:(batch+1)*max_examples], bottleneck)
                    acts = mymodel.reshape_activations(acts).squeeze()
                    class_id = mymodel.label_to_id(item['class'].split(',')[0])
                    grad = mymodel.get_gradient(acts, [class_id], bottleneck)
                    reshaped = np.reshape(grad, -1)
                    acts_all.append(acts)
                    grads_all.append(grad)
               
                    with tf.gfile.Open(wd+item['id']+'_sorted_acts', 'w') as f:
                      np.save(f, acts_all, allow_pickle=False) 
                    with tf.gfile.Open(wd+item['id']+'_sorted_grads', 'w') as f:
                      np.save(f, grads_all, allow_pickle=False) 
                except Exception as e:
                    print("Problem with class: {} with id: {}".format(item["class"],item['id']))
                    print(e)
            delta = datetime.now() - start_time 
            print("Class: {1} activations and gradients collected (id: {0}) took: {2} | remaining time estimate: {3}".format(item['id'],item['class'],delta, delta*(len(classes)-i+1)))


    print("DONE!")


if __name__ == '__main__':
        main()

