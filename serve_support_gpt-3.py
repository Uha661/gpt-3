import tensorflow as tf
from tensorflow import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_constants import PREDICT_METHOD_NAME
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model.signature_def_utils import build_signature_def
import tensorflow as tf
import sys


# tensor list is used to print the list of tensors, whick can be used to find the exact names of the tensor lists given 
def tensor_list(graph_location):
    with tf.Session(graph=tf.get_default_graph()) as sess:

        saver = tf.train.import_meta_graph(graph_location+'.meta')
        saver.restore(sess, graph_location)

        for op in tf.get_default_graph().get_operations():
           print(str(op.values()), file=open("Tensorlist.txt", "a"))


    return('done with tensor_list please check it')



def add_tags(graph_location,save_location):

    with tf.Session(graph=tf.get_default_graph()) as sess:
        
        saver = tf.train.import_meta_graph(graph_location+'.meta')
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, graph_location)


        
        builder = tf.saved_model.Builder(save_location)
  
        input_tensor   = tf.get_default_graph().get_tensor_by_name("sample_sequence/strided_slice/stack:0")
        output_tensor  = tf.get_default_graph().get_tensor_by_name("sample_sequence/while/Exit_4:0")
  

        tensor_info_x = tf.saved_model.utils.build_tensor_info(input_tensor)
         # lets look where input_tensor is defined
        tensor_info_y = tf.saved_model.utils.build_tensor_info(output_tensor)
        
        signature = build_signature_def(inputs={'x_input': tensor_info_x},outputs={'y_output': tensor_info_y},method_name=PREDICT_METHOD_NAME)
        

sample_sequence/strided_slice/stack:0
        
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})


        builder.save()


    return print('added_tags')





## command line argument smaple is like this
## python serve_support.py "task_name" "Location_of_graph"  "Location_to_save_model"
## python serve_support.py add_tags ../models/apr15/-2323  ../models/apr16/save_model/1

arg=sys.argv
task_name=arg[1]
if (task_name=="tensor_list"):
  Location_of_graph=arg[2]
  print(tensor_list(Location_of_graph))
if(task_name=="add_tags"):
  Location_of_graph=arg[2]
  Location_to_save_model=arg[3]
  add_tags(Location_of_graph,Location_to_save_model)


