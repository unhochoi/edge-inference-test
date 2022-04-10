# 실제로는 작동되지 않는 코드 (기록용)

# import tensorflow as tf
# from transformers import TFAutoModelForSequenceClassification

# TF 기반 NLP 모델
# origin_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# GLUE SST-2 로 학습된 DistilBERT 의 경우, 문장의 최대 길이가 128
# max_seq_length = 128

# 모델의 입출력 형식을 지정해주는 것 같은데.. 데이터에 따라 달라질 수 있는지 추가적인 조사가 필요할 듯
# input_spec = tf.TensorSpec([None, max_seq_length], tf.int32)
# origin_model._set_inputs(input_spec, training=False)

# 변수로 저장되어 있는 모델을, TFLite 모델로 변환
# converter = tf.lite.TFLiteConverter.from_keras_model(origin_model)

# FP16 양자화 설정
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_types = [tf.float16]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter = True

# 모델 
# tflite_fp16_model = converter.convert()

# 변환된 모델을 .tflite 파일에 저장
# open("distilbert_convert_tflite_fp16_model.tflite", "wb").write(tflite_fp16_model)

# 파일로 저장되어 있는 모델을 Interpreter 로 로드
# interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_fp16_model))

# 변수에 저장되어 있는 모델을 Interpreter 로 로드
# interpreter_fp16 = tf.lite.Interpreter(tflite_fp16_model)
# interpreter_fp16.allocate_tensors()

# 간단한 문장을 통해 변환한 모델로 추론 진행
# test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

# input_index = interpreter.get_input_details()[0]["index"]
# output_index = interpreter.get_output_details()[0]["index"]

# interpreter.set_tensor(input_index, test_image)

# inference 실행
# interpreter.invoke()


# predictions = interpreter.get_tensor(output_index)
