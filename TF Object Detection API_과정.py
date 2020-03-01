TensorFlow Object Detection API

    <Ubuntu 16.04>
        1. https://www.ubuntu.com/
            > [Downloads] - [Desktop] - Ubuntu 16.04.3 LTS
        2. Universal USB Installer
        3. 파티션
            > swap : 32GB (주파티션, 스왑)
              /    : 50GB (주파티션, ex4)
              /boot: 50mb (주파티션, ex4)
              /var : 5GB  (논리파티션, ex4)
              /database : 150GB (논리파티션, ex4)
              /home : 나머지메모리 (논리파티션, ex4)

    <TensorFlow with GPU 설치>
        0. Version
            > CUDA Toolkit 8.0
            > cuDNN SDK v6.0
            > Anaconda 3

        1. Nvidia 그래픽 드라이버 설치
            > $ sudo add-apt-repository ppa:graphics-drivers/ppa
              $ sudo apt-get update
              $ sudo apt-get install nvidia-375
              $ sudo reboot

            > [시스템 설정] - [스프트웨어 & 업데이트] - [추가 드라이버]

        2. CUDA Toolkit 설치
            > CUDA Toolkit 8.0 설치
            > https://developer.nvidia.com/cuda-80-ga2-download-archive
                : [Linux] - [x86_64] - [Ubuntu] - [16.04] - [runfile(local)]
                : $ sudo sh cuda_8.0.61_375.26_linux.run
            > 환경변수
                : $ sudo gedit ~/.bashrc
                  export CUDA_HOME=/usr/local/cuda-8.0
                  export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
                  export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
                : $ source ~/.bashrc
            > 설치확인
                : $ nvcc --version
            > 삭제
                : $ sudo apt-get remove --auto-remove nvidia-cuda-toolkit

        3. cuDNN 설치
            > th_k9304@naver.com / Rlaxogud753465
            > https://developer.nvidia.com/rdp/cudnn-archive
            > Download cuDNN v6.0[April 27, 2017], for CUDA 8.0
                : $ tar xvzf cudnn-8.0-linux-x64-6.0.tgz
                  $ sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
                  $ sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
                  $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

                : $ tar xvzf cudnn-8.0-linux.....
                  $ sudo mv cuda/include/cudnn.h /usr/local/cuda-8.0/include
                  $ sudo mv cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64
                  $ sudo chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*

        4. pip Update
            > $ sudo apt-get install python3-pip
              $ sudo apt-get install python-pip
              $ pip install -upgrade pip
              $ pip3 install -upgrade pip

        5. Anaconda
            > https://www.anaconda.com/download/#linux
                : [How to get Python 3.5 or other Python versions] - [Anaconda installer archive]
                    - 'Anaconda3-4.2.0-Linux-x86_64.sh'
                : $ cd 다운받은폴더
                  $ bash Anaconda3-4.2.0-Linux-x86_64.sh

                  $ sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
                  $ sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
                  $ sudo update-alternatives --config python3
                        : 0 -> enter
                  $ source ~/.bashrc

            > 가상환경 생성
                : $ conda create - n Kimtae python=3.5 anaconda
                  $ conda info --envs
                  $ source activate Kimtae

            > TensorFlow 설치
                : (Kimtae)$ sudo apt-get install python3-pip python3-dev
                  (Kimtae)$ pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp35-cp35m-linux_x86_64.whl
                  (Kimtae)$ pip3 install tensorflow-gpu

    <TensorFlow Object Detection API>
        [요약]
            - Google Protocol Buffer
            - .config file
            - .record file
            - .pbtxt / .xlsx

        Tensorflow Object Detection API depends on the following libraries:
            > Protobuf 3+
            > python-tk
            > Pillow 1.0
            > lxml
            > tf Slim (which is included in the "tensorflow/models/research/" checkout)
            > Jupyter notebook
            > Matplotlib
            > TensorFlow
            > Cython
            > cocoapi

            : $ sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
              $ sudo pip install Cython
              $ sudo pip install jupyter
              $ sudo pip install matplotlib

        1. Protobuf Compilation
            - record 파일을 사용하기 위해
            : # From tensorflow/models/research/
              $ protoc object_detection/protos/*.proto --python_out=.

              # From tensorflow/models/research/
              $ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
              # 고정시켜 놓으려면 ~/.bashrc에 적자

              # Testing the Installation
              $ python object_detection/builders/model_builder_test.py

        2. .pbtxt / .xlsx
            - nose : 1
              mouth : 2
            - my_label_map.pbtxt
                : item {
                    name: "nose"
                    id: 1
                    display_name: "nose"
                  }
                  item {
                    name: "mouth"
                    id: 2
                    display_name: "mouth"
                  }
            - my_label_map.xlsx

        3. .TFRecord file
            - tensorflow에서 training 할 때 사용하는 파일
            - tensorflow에서 사용하는 파일 형식
            - (이미지 + label 파일)을 binary 파일(.record)로 변환
            : # From tensorflow/models/research/
              $ python object_detection/make_tfrecord_allclassify.py \
                   --inputpath=/home/Kimtae/Dataset/thermo-graphic/training \
                   --outfilename=/home/Kimtae/Dataset/thermo-graphic/train.record \
                   --labelmappath=/home/Kimtae/Dataset/thermo-graphic/my_label_map.xlsx
{
            """
            python make_tfrecord_semiclassify.py
                --inputpath [str]
                --outfilename [.record]
                --labelmappath [.xlsx]
            """

            from xml.etree.ElementTree import parse
            import os
            import io
            import hashlib
            import tensorflow as tf
            import PIL.Image
            import argparse
            import xlrd
            import sys

            # sys.path.append('/home/prlab-ubuntu/TF_Utiles/models/research')
            # sys.path.append('/home/prlab-ubuntu/TF_Utiles/models/research/slim')

            from utils import dataset_util

            # argument check
            parser = argparse.ArgumentParser()
            parser.add_argument('--inputpath', default='/home/prlab-ubuntu/Kimtae/Dataset/thermo-graphic/training', type=str, required=False)
            parser.add_argument('--outfilename', default='/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training/train2.record',type=str, required=False)
            parser.add_argument('--labelmappath', default='/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training/my_label_map.xlsx', type=str, required=False)
            args = parser.parse_args()

            def convert_from_name_to_label(class_name, label_map):
                label = 0
                valid = False

                for item in label_map:
                    if item[0] == class_name:
                        label = int(item[1])
                        break

                return label


            def main():
                workbook = xlrd.open_workbook(args.labelmappath)
                work_sheet = workbook.sheet_by_index(0)

                num_of_classes = work_sheet.nrows

                label_map = []
                for idx in range(0,num_of_classes):
                    label_map.append(work_sheet.row_values(idx))

                annotations = list()

                for (path, dir, files) in os.walk(args.inputpath):
                    for filename in files:
                        ext = os.path.splitext(filename)[-1]
                        if ext == '.xml':
                            annotations.append(filename)

                writer = tf.python_io.TFRecordWriter(args.outfilename)

                for i, annotation in enumerate(annotations):
                    print("%d/%d --- %s/%s" % (i, len(annotations), args.inputpath, annotation))

                    # read image file
                    xml_full_path = args.inputpath + "/" + annotation
                    image_path = xml_full_path.replace(".xml", ".jpg")

                    with tf.gfile.GFile(image_path, 'rb') as fid:
                        encoded_jpg = fid.read()
                    encoded_jpg_io = io.BytesIO(encoded_jpg)
                    image = PIL.Image.open(encoded_jpg_io)
                    width, height = image.size

                    if image.format != 'JPEG':
                        raise ValueError('Image format not JPEG')
                    key = hashlib.sha256(encoded_jpg).hexdigest()

                    image_file_name = annotation.replace(".xml", ".jpg")

                    # read xml file
                    xml_tree = parse(xml_full_path)
                    xml_root = xml_tree.getroot()

                    objects = xml_root.findall("object")

                    xmins = []
                    xmaxs = []
                    ymins = []
                    ymaxs = []
                    class_names = []
                    class_labels = []

                    num_of_objects = 0

                    for object in objects:
                        class_name = object.find("name").text
                        class_label = convert_from_name_to_label(class_name, label_map)

                        if class_label > 0:
                            num_of_objects += 1

                            bounding_box = object.find("bndbox")
                            xmin = float(bounding_box.find("xmin").text)
                            ymin = float(bounding_box.find("ymin").text)
                            xmax = float(bounding_box.find("xmax").text)
                            ymax = float(bounding_box.find("ymax").text)

                            xmins.append(float(xmin) / width)
                            xmaxs.append(float(xmax) / width)
                            ymins.append(float(ymin) / height)
                            ymaxs.append(float(ymax) / height)

                            class_names.append(class_name.encode('utf8'))
                            class_labels.append(class_label)

                    if num_of_objects > 0:
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image/height': dataset_util.int64_feature(height),
                            'image/width': dataset_util.int64_feature(width),
                            'image/filename': dataset_util.bytes_feature(image_file_name.encode('utf8')),
                            'image/source_id': dataset_util.bytes_feature(image_file_name.encode('utf8')),
                            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                            'image/object/class/text': dataset_util.bytes_list_feature(class_names),
                            'image/object/class/label': dataset_util.int64_list_feature(class_labels),
                        }))
                        writer.write(example.SerializeToString())


                writer.close()

            if __name__ == "__main__":
                main()
}

        4. .config file
            - faster_rcnn_resnet101.config
            - 이 file에서 .record file, .pbtxt file의 입력 경로를 지정 해준다.
{
    # Faster R-CNN with Resnet-101 (v1), configuration for MSCOCO Dataset.
    # Users should configure the fine_tune_checkpoint field in the train config as
    # well as the label_map_path and input_path fields in the train_input_reader and
    # eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
    # should be configured.

    model {
    faster_rcnn {
    num_classes: 2
    image_resizer {
    keep_aspect_ratio_resizer {
    min_dimension: 230
    max_dimension: 330
    }
    }
    feature_extractor {
    type: 'faster_rcnn_resnet101'
    first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
    grid_anchor_generator {
    #height : 50
    #width : 50
    scales: [0.25, 0.5, 1.0, 2.0]
    aspect_ratios: [0.5, 1.0, 2.0]
    height_stride: 16
    width_stride: 16
    }
    }
    first_stage_box_predictor_conv_hyperparams {
    op: CONV
    regularizer {
    l2_regularizer {
    weight: 0.0
    }
    }
    initializer {
    truncated_normal_initializer {
    stddev: 0.01
    }
    }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
    mask_rcnn_box_predictor {
    use_dropout: false
    dropout_keep_probability: 1.0
    fc_hyperparams {
    op: FC
    regularizer {
    l2_regularizer {
    weight: 0.0
    }
    }
    initializer {
    variance_scaling_initializer {
    factor: 1.0
    uniform: true
    mode: FAN_AVG
    }
    }
    }
    }
    }
    second_stage_post_processing {
    batch_non_max_suppression {
    score_threshold: 0.0
    iou_threshold: 0.6
    max_detections_per_class: 100
    max_total_detections: 300
    }
    score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
    }
    }

    train_config: {
    batch_size: 1
    optimizer {
    momentum_optimizer: {
    learning_rate: {
    manual_step_learning_rate {
    initial_learning_rate: 0.0003
    schedule {
    step: 900000
    learning_rate: .00003
    }
    schedule {
    step: 1200000
    learning_rate: .000003
    }
    }
    }
    momentum_optimizer_value: 0.9
    }
    use_moving_average: false
    }
    gradient_clipping_by_norm: 10.0
    #fine_tune_checkpoint: "/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training/#model.ckpt"
    #from_detection_checkpoint: true
    #data_augmentation_options {
    #  random_horizontal_flip {
    #  }
    #}


    train_input_reader: {
    tf_record_input_reader {
    input_path: "/home/Kimtae/Dataset/thermo-graphic/train.record"
    }
    label_map_path: "/home/Kimtae/Dataset/thermo-graphic/my_label_map.pbtxt"
    }

    eval_config: {
    num_examples: 8000
    # Note: The below line limits the evaluation process to 10 evaluations.
    # Remove the below line to evaluate indefinitely.
    max_evals: 10
    }

    eval_input_reader: {
    tf_record_input_reader {
    input_path: "/home/Kimtae/Dataset/thermo-graphic/valid.record"
    }
    label_map_path: "/home/Kimtae/Dataset/thermo-graphic/my_label_map.pbtxt"
    shuffle: false
    num_readers: 1
    }
    }
}

        5. Training
            : # Fram tensorflow/models/research/
              $ nohup python object_detection/train.py \
                   --logtostderr \
                   --train_dir=object_detection/training/model/ \
                   --pipeline_config_path=object_detection/training/faster_rcnn_resnet101.config

            : $ tensorboard --logdir='모델 저장 경로'

        6. Testing
            - .ckpt file을 .pb file로 변환 해줘야 한다.

            : # From tensorflow/models/research/
              $ python object_detection/export_inference_graph.py \
                   --input_type image_tensor \
                   --pipeline_config_path='모델 저장 경로/pipeline.config' \
                   --trained_checkpoint_prefix='모델 저장 경로/model.ckpt-nnnn' \
                   --output_directory='모델 저장 경로/output_inference_graph'

            - 동영상에 적용
            : $ python object_detection_tutorial.py
{

    # coding: utf-8

    # # Object Detection Demo
    # Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

    # # Imports

    # In[ ]:


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

    #++
    import cv2

    Video_input_Path = '/home/prlab-ubuntu/Kimtae/Dataset/thermo-graphic/Video/sgh.avi'
    Video_save_Path = '/home/prlab-ubuntu/Kimtae/Dataset/thermo-graphic/testing/model2/sgh.avi'

    cap = cv2.VideoCapture(Video_input_Path)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    output = cv2.VideoWriter(Video_save_Path, fourcc, 20.0, (320, 240))

    if not cap:
        print('Error')
        sys.exit(1)

    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append("..")
    from object_detection.utils import ops as utils_ops

    if tf.__version__ < '1.4.0':
      raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


    # ## Env setup

    # In[ ]:


    # This is needed to display the images.
    # --
    #get_ipython().run_line_magic('matplotlib', 'inline')


    # ## Object detection imports
    # Here are the imports from the object detection module.

    # In[ ]:


    from utils import label_map_util

    from utils import visualization_utils as vis_util


    # # Model preparation

    # ## Variables
    #
    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
    #
    # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = '/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training/model/output_inference_graph/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training/my_label_map.pbtxt'
    NUM_CLASSES = 1


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    # ## Helper code

    # In[ ]:


    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    # # Detection

    # In[ ]:


    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = '/home/prlab-ubuntu/TF_Utiles/models/research/object_detection/training/valid/'
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 4) ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)


    # In[ ]:


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
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict


    # In[ ]:

    #--
    #for image_path in TEST_IMAGE_PATHS
    #++
    while True:
      #--
      #image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      #image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      #++
      ret, image_np = cap.read()
      if not ret:
          print('Video end')
          break
      output_dict = run_inference_for_single_image(image_np, detection_graph)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)

      cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      output.write(image_np)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          cap.release()
          output.release()
          cv2.destroyAllWindows()
          break
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)

}
