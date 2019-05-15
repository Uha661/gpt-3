import tensorflow as tf
import json
import numpy as np
from grpc.beta import implementations
import grpc
import numpy
import sys
from datetime import datetime
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import model, encoder

_counter = 0
_start = 0


enc = encoder.get_encoder('117M')


def do_inference(hostport,input_string):


  # create connection
  host, port = hostport.split(':')
  #channel = implementations.insecure_channel(host, int(port))
  channel = grpc.insecure_channel(hostport)
  # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
 
  # initialize a request
  request = predict_pb2.PredictRequest()
  # Name of the model that is running on docker
  request.model_spec.name = 'saved_model'
  # name of the signatures that we defined in builder_tags
  request.model_spec.signature_name = 'serving_default'
  #data

  ids=enc.encode('hello')

 

# reshaping the input tensors as defined in signature defnition
  print(len(ids))

  tensor = tf.contrib.util.make_tensor_proto(ids, dtype=tf.int32,shape=(1,1))
  


  request.inputs['x_input'].CopyFrom(tensor)
  
  result = stub.Predict(request, 15.0) 

  

 
  return result
 


## Command line arguments 
## pyhton client.py "input_mode" 
## python client.py server 

arg=sys.argv
input_mode=arg[1]


if input_mode=='server':
  ip='192.168.2.210:8500'
if input_mode=='local':
  ip='10.0.75.1:8500'



input_data=open('gpt-3_test_input.json')
inputs = json.load(input_data)
input_data.close()




for input_words in inputs:
  start_time = time.time()

  result     = do_inference(hostport=ip,input_string=input_words)

  print((time.time() - start_time))



  count=0

  print(result.outputs["y_output"])
  break
  print("'")
  print("'")
  print("'")
  print("'")
  print("'")
  print("'")
  print("'")

  #result.ouputs["w_output"] has indices of top 10 predictions which can be refered back in the lookup table

  for  val in result.outputs["y_output"].int_val:
     print(input_words+' ' + enc.decode(val))
     #print(result.outputs["y_output"].float_val[count])
     count=count+1
