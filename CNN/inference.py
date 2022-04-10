# 실제 동작 여부 및 오류 파악이 안된 코드입니다.


# 추론에 필요한 library import
import os
import time
import numpy as np
import shutil
import requests
from functools import partial
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# tensorflow 에서 제공하는 model import
from tensorflow.keras.applications import ( 
        mobilenet,
        mobilenet_v2,
        inception_v3
        )

# models = {
#         'mobilenet':mobilenet,
#         'mobilenet_v2':mobilenet_v2,
#         'inception_v3':inception_v3
#         }

# Imagenet 추론에 최적화 되어있는 모델 사용
models_detail = {
        'mobilenet':mobilenet.MobileNet(weights='imagenet'),
        'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
        'inception_v3':inception_v3.InceptionV3(weights='imagenet')
        }

# 원하는 모델을 library 에서 import 한 뒤, Local에 저장하는 함수 (모델명, 저장파일명)
def load_save_model(model, saved_model_dir = 'mobilenet_saved_model'):
    # Imagenet 추론에 최적화 되어있는 모델 생성
    model = models_detail[model]
    # Local에 존재하는 기존 모델을 삭제
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    # Local에 모델 저장
    model.save(saved_model_dir, include_optimizer=False, save_format='tf')

# tfrecord 파일 내부의 인코딩 정보를 가져오는 함수
def deserialize_image_record(record):

    # tfrecord 파일 내부의 인코딩된 값, 라벨, 라벨명을 별도의 변수로 저장
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                  'image/class/text': tf.io.FixedLenFeature([], tf.string, '')}
    
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    label_text = tf.cast(obj['image/class/text'], tf.string)   
    
    return imgdata, label, label_text

# 인코딩 된 tfrecord를 디코딩 하는 함수
def val_preprocessing(record):
    
    # 인코딩 된 tfrecord의 정보
    imgdata, label, label_text = deserialize_image_record(record)
    label -= 1
    
    # tfrecord의 정보를 기반으로, 다시 jpeg로 디코딩
    image = tf.io.decode_jpeg(imgdata, channels=3, 
                              fancy_upscaling=False, 
                              dct_method='INTEGER_FAST')
    # 디코딩한 이미지 형태 및 값 변환
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    # 0~255의 화소를 scale
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)
    scale = tf.cond(tf.greater(height, width),
                  lambda: side / width,
                  lambda: side / height)
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
    
    # 이미지 shape를 변경 
    # 배치 단위를 None으로 추가 -> (None, ?, ?, ?)
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    
    return image, label, label_text

# Local에 저장되어 있는 데이터를 가져온 뒤, 데이터 shape에 batch 단위를 추가하는 함수
def get_dataset(batch_size):
    # 데이터 경로
    data_dir = './validation-00000-of-00001'
    # 데이터 경로에 존재하는 파일 반환
    files = tf.io.gfile.glob(os.path.join(data_dir))
    # 이미지 데이터를 TFRecord 형식으로 변환 (원본 이미지를 .tfrecord 라는 바이너리 데이터 포맷으로 압축하는 것)
    dataset = tf.data.TFRecordDataset(files)

    # 압축된 파일을 디코딩 (배치 단위 추가)
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 데이터 shape에 배치 사이즈 할당
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset
    

def calibrate_fn(n_calib, batch_size, dataset):
    for i, (calib_image, _, _) in enumerate(dataset):
        if i > n_calib // batch_size:
            break
        yield (calib_image,)


# TensorRT 에서 양자화 방법에 따라 모델을 컴파일하는 함수
def build_FP_tensorrt_engine(model, quantization, batch_size):

    #dataset = get_dataset(batch_size)

    if quantization == 'FP32':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP32,
                                                        maximum_cached_engines=num_engines,
                                                        max_workspace_size_bytes=8000000000)
    elif quantization == 'FP16':                                                 
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.FP16,
                                                        maximum_cached_engines=num_engines,
                                                        max_workspace_size_bytes=8000000000)
    
    elif quantization == 'INT8':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
                                                        precision_mode=trt.TrtPrecisionMode.INT8, 
                                                        max_workspace_size_bytes=8000000000, 
                                                        maximum_cached_engines=num_engines,
                                                        use_calibration=True)

    
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=f'{model}_saved_model',
                                        conversion_params=conversion_params)
    
    if quantization=='INT8':
        n_calib=50
        converter.convert(calibration_input_fn=partial(calibrate_fn, n_calib, batch_size, dataset.shuffle(buffer_size=n_calib, reshuffle_each_iteration=True)))

    else:
        converter.convert()
        
    trt_compiled_model_dir = f'{model}_saved_models_{quantization}'
    converter.save(output_saved_model_dir=trt_compiled_model_dir)

    return trt_compiled_model_dir

# Local에 저장된 모델을 사용해, 배치 단위로 이미지 추론
def predict_tf(batch_size,saved_model_dir):
    
    # Local에 저장되어 있는 model을 load하는 시간
    model_load_time = time.time()
    model = tf.keras.models.load_model(saved_model_dir)
    model_load_time = time.time() - model_load_time
    
    # 이미지 5000개 마다 결과를 보기위한 디버깅용 변수
    display_every = 5000
    display_threshold = display_every
    
    # 전체 이미지에 대한 예측라벨 및 실제라벨 저장
    pred_labels = []
    actual_labels = []

    # 배치 단위에 따른 추론 시간 저장
    iter_times = []
    
    # Local에 저장되어 있는 data를 load하는 시간 (data -> decoding -> encoding)
    dataset_load_time = time.time()
    dataset = get_dataset(batch_size)
    dataset_load_time = time.time() - dataset_load_time
    
    # 전체 이미지에 대해 추론 시작
    iftime_start = time.time()
    # 이미지 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        
        # 배치 단위별 이미지 추론이 시작한 시간
        start_time = time.time()
        # 배치 단위별 데이터셋 분류 (결과 : 이미지 1장에 대한 클래스 별 유사도가, 배치 사이즈 만큼 존재)
        pred_prob_keras = model(validation_ds)
        # 배치 단위별 이미지 추론이 끝난 시간
        iter_times.append(time.time() - start_time)
        
        # 배치 사이즈 만큼의 실제 라벨 
        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        # 배치 사이즈 만큼의 예측 라벨 (클래스 별 유사도에서 가장 유사도가 높은 값으로 예측)
        pred_labels.extend(list(np.argmax(pred_prob_keras, axis=1)))
        # 디버깅 용 코드
        # if i*batch_size >= display_threshold:
        #     display_threshold+=display_every
    
    # 모든 이미지에 대한 배치별 추론 시간을 배열화
    iter_times = np.array(iter_times)

    # 모든 이미지에 대한 실제라벨과 예측라벨을 비교한 뒤, 정확도 계산
    acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    
    # 전체 이미지 추론에 대한 결과
    print('***** TF-FP32 metric *****')
    print('user_batch_size =', batch_size)
    print('accuracy =', acc_keras_gpu)   
    print('model_load_time =', model_load_time)
    print('dataset_load_time =', dataset_load_time)
    
    # 전체 이미지 추론 시간
    print('inference_time =', time.time() - iftime_start)
    
    # 배치별 추론의 평균 시간 (배치별 이미지 추론 시간의 합 / 배치횟수)
    print('inference_time(avg) =', np.sum(iter_times)/len(iter_times))
    
    # 추가 수정 필요
    # 배치횟수 / (모델 로드 시간 + 데이터셋 로드 시간 + 전체 이미지 추론 시간)
    # print('IPS =', len(iter_times) / (model_load_time + dataset_load_time + (time.time() - iftime_start)))
    
    # 이미지개수 / 전체 배치별 추론 시간의 합 = 1초에 몇장의 이미지를 처리하는지
    print('IPS(inf) =', (len(iter_times)*batch_size) / np.sum(iter_times))    
    # 이미지개수 30장 (10장,10장,10장)
    # 배치별 추론 시간의 합은 3초 (1초, 1초, 1초)
    # 30/3 = 10
    # 1초당 10장 처리

# Local에 저장된 모델을 사용해, 배치 단위와 양자화 방법에 따라 이미지 추론
def predict_trt(trt_compiled_model_dir, quantization, batch_size):

    # Local에 저장되어 있는 model을 load하는 시간
    model_load_time = time.time()
    saved_model_trt = tf.saved_model.load(trt_compiled_model_dir, tags=[tag_constants.SERVING])
    model_trt = saved_model_trt.signatures['serving_default']
    model_load_time = time.time() - model_load_time

    # 이미지 5000개 마다 결과를 보기위한 디버깅용 변수    
    display_every = 5000
    display_threshold = display_every

    # 전체 이미지에 대한 예측라벨 및 실제라벨 저장
    pred_labels = []
    actual_labels = []

    # 배치 단위에 따른 추론 시간 저장
    iter_times = []
    
    # Local에 저장되어 있는 data를 load하는 시간 (data -> decoding -> encoding)
    dataset_load_time = time.time()
    dataset = get_dataset(batch_size)
    dataset_load_time = time.time() - dataset_load_time

    # 전체 이미지에 대해 추론 시작
    iftime_start = time.time()
    # 이미지 데이터를 배치 단위로 묶어서 사용 (반복문 한번당 배치 단위 추론 한번)
    for i, (validation_ds, batch_labels, _) in enumerate(dataset):
        # 배치 단위별 이미지 추론이 시작한 시간
        start_time = time.time()
        # 배치 단위별 데이터셋 분류 (결과 : 이미지 1장에 대한 클래스 별 유사도가, 배치 사이즈 만큼 존재)
        trt_results = model_trt(validation_ds)
        # 배치 단위별 이미지 추론이 끝난 시간
        iter_times.append(time.time() - start_time)

        # 배치 사이즈 만큼의 실제 라벨 
        actual_labels.extend(label for label_list in batch_labels.numpy() for label in label_list)
        # 배치 사이즈 만큼의 예측 라벨 (클래스 별 유사도에서 가장 유사도가 높은 값으로 예측)
        pred_labels.extend(list(tf.argmax(trt_results['predictions'], axis=1).numpy()))
        # 디버깅용 코드
        # if i*batch_size >= display_threshold:
        #    display_threshold+=display_every

    iter_times = np.array(iter_times)
    acc_keras_gpu = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)    

    print('***** TRT-quantization metric *****')
    # 전체 이미지 추론에 대한 결과
    print('user_batch_size =', batch_size)
    print('accuracy =', acc_keras_gpu)   
    print('model_load_time =', model_load_time)
    print('dataset_load_time =', dataset_load_time)
    
    # 전체 이미지 추론 시간
    print('inference_time =', time.time() - iftime_start)
    
    # 배치별 추론의 평균 시간 (배치별 이미지 추론 시간의 합 / 배치횟수)
    print('inference_time(avg) =', np.sum(iter_times)/len(iter_times))
    
    # Inference 의 end-to-end 작업 시간을 모두 고려했을 때, 1초에 추론 가능한 데이터 개수
    # 이미지개수 / (모델 로드 시간 + 데이터셋 로드 시간 + 전체 이미지 추론 시간)
    print('IPS =', (len(iter_times)*batch_size) / (model_load_time + dataset_load_time + (time.time() - iftime_start)))
    
    # 단순히, Inference 시간만 고려했을 때, 1초에 추론 가능한 데이터 개수
    # 이미지개수 / 전체 배치별 추론 시간의 합
    print('IPS(inf) =', (len(iter_times)*batch_size) / np.sum(iter_times))    

# 명령어로 인자들을 입력받기 위한 설정
results = None
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', default=1, type=int)
parser.add_argument('--model', default='mobilenet', type=str)
parser.add_argument('--case', type=str, required=True)
parser.add_argument('--quantization',default='FP32',type=str)
parser.add_argument('--engines',default=1, type=int)
parser.add_argument('--img_size',default=224, type=int)
args = parser.parse_args()

# 명령어로 입력 받은 인자들을 변수로 할당
batch_size = args.batchsize
model = args.model
case=args.case
quantization = args.quantization
num_engines=args.engines
img_size=args.img_size

# Local에 저장할 모델명
saved_model_dir = f'{model}_saved_model'

# 원하는 모델을 library 에서 import 한 뒤, Local에 저장 (모델명, 저장파일명)
load_save_model(model, saved_model_dir)

# 플랫폼에 따라 최적화 및 추론

# TensorFlow로 추론
if case == 'tf' :
    # Local에 저장된 모델을, 배치 사이즈에 따라 추론
    predict_tf(batch_size,saved_model_dir)

# TensorRT로 추론
elif case == 'trt' :
    # 모델을, 배치사이즈와 양자화 방법에 따라 컴파일
    trt_compiled_model_dir = build_FP_tensorrt_engine(model, quantization, batch_size)
    
    # Local에 저장된 모델을, 양자화 방법과 배치 사이즈에 따라 추론
    predict_trt(trt_compiled_model_dir, quantization, batch_size)
