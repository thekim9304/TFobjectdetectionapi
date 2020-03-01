
# coding: utf-8

# # Import

# In[198]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import csv
import xml.etree.ElementTree as parse

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Function for Evaluation

# In[199]:


def parse_xml(xml_path):


    tree = parse.parse(xml_path)
    obj = tree.getroot()

    bndbox = []

    for object in obj.findall('object'):
        if object.find('name').text == 'nose':
            for x in object.find('bndbox'):
                bndbox.append(x.text)
                # xmin, ymin, xmax, ymax

    size = obj.findall('size')
    width = size[0].find('width').text
    height = size[0].find('height').text

    gndTruth = []

    ymin = int(bndbox[1]) / int(height)
    xmin = int(bndbox[0]) / int(width)
    ymax = int(bndbox[3]) / int(height)
    xmax = int(bndbox[2]) / int(width)

    gndTruth.append(ymin)
    gndTruth.append(xmin)
    gndTruth.append(ymax)
    gndTruth.append(xmax)

    return gndTruth


# In[200]:


def calcu_IoU(gndTruth, Bbox):
    gymin = float(gndTruth[0])
    gxmin = float(gndTruth[1])
    gymax = float(gndTruth[2])
    gxmax = float(gndTruth[3])

    pymin = float(Bbox[0])
    pxmin = float(Bbox[1])
    pymax = float(Bbox[2])
    pxmax = float(Bbox[3])

    A1 = (gxmax - gxmin) * (gymax - gymin)
    A2 = (pxmax - pxmin) * (pymax - pymin)
    inter = (min(gxmax, pxmax) - max(gxmin, pxmin)) * (min(gymax, pymax) - max(gymin, pymin))
    union = A1 + A2 - inter

    IoU = inter / union

    return IoU


# In[201]:


def extract_corrdinate(Bbox):
    corrdi = []

    for corrdinate, classes in Bbox.items():
        if classes[0][:4] == 'nose':
            corrdi.append(corrdinate)

    return corrdi


# In[202]:


def evaluate_image(BboxAll, image_name):
    threshold = 0.5

    gndTruth = parse_xml(image_name)
    Bbox = extract_corrdinate(BboxAll)

    write_imageInfo = []
    write = []

    write_imageInfo.append(image_name)
    write_imageInfo.append(len(Bbox))

    for i in range(len(Bbox)):
        IoU = calcu_IoU(gndTruth, Bbox[i])
        if IoU >= threshold:
            write_imageInfo.append('true')
        else:
            write_imageInfo.append('false')
        write_imageInfo.append(IoU)

    return write_imageInfo


# # Path Option

# In[203]:


PATH_TO_CKPT = '/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training_second/models_2/frozen_inference_graph.pb'

PATH_TO_LABELS = '/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/my_label_map.pbtxt'
NUM_CLASSES = 2

PATH_TO_TEST_IMAGES_DIR = '/home/prlab-ubuntu/Kimtae/testImage/'

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(0, 4)]


# # Detection

# In[204]:


for thr in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1]:
    write = []

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    IMAGE_SIZE = (12,8)

    def run_inference_for_single_image(image, graph):
      with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      # Visualization of the results of a detection.
      Bbox = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)

      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)

      xml_path = image_path[:-3] + 'xml'


      threshold = thr

      write.append(evaluate_image(Bbox, xml_path, threshold))

    csv_name = 'nose_' + str(thr) + '.csv'

    print(thr, csv_name)

    f = open(csv_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    for i in range(len(write)):
        wr.writerow(write[i])

    f.close()


# # Write on csv

# In[205]:


f = open('daat3.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)

for i in range(len(write)):
    wr.writerow(write[i])

f.close()


# # ================================================
# # ================================================
