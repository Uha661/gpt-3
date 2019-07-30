

import fire
import json
import time 
import os
import numpy as np
import tensorflow as tf
import json
import model, sample_spoken_edit, encoder


def Test_and_save_simple_graph(input_test_file=None,Original=True, model_name='117M', length=1, temperature=1, top_k=10 ):

    # A json file that has all the input test words
    test_input=open(input_test_file)
    inputs_words = json.load(test_input)
    test_input.close()
    

    start_time = time.time()
    # enc is the encoder from the model
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))



    with tf.Session(graph=tf.Graph()) as sess:
        tf.enable_resource_variables()
        # i would add thi shere beacuse our input X is intiated right below it.

        context = tf.placeholder(tf.int32, [1, None])
        
        # This intilaises the a small model only to test, but this model still has same trained weights used in the trained model. 
        output = sample_spoken_edit.sample_sequence(
            hparams=hparams, length=1,
            context=context,
            temperature=temperature, top_k=top_k
        )
        
        
        saver = tf.train.Saver()
        
        # Please make sure that we are passing the checkpoints of trained model 
        if(Original):
            ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name)) 
        else: 
            ckpt = tf.train.latest_checkpoint(os.path.join('checkpoint', 'run1'))
        
        # Restores the graph from saved model 
        saver.restore(sess, ckpt)
        
        

        

        # saves the test graph which is smaller and only has sample sequence process in it.
        saver.save(sess,os.path.join('checkpoint-test', 'run1', 'model-test'))
        print('saving the model at checkpoint-test/run1/model-test')







        # this print is to know time taken to load model 
        print(str(round((time.time() - start_time)*1000, 1))+' time to intialise model in milli Sec')


        for raw_text in inputs_words:
            start_time = time.time()
            context_tokens = enc.encode(raw_text)
            print()

            print('"""""""')
            print('"""""""')
            print('"""""""')
            print(context_tokens)
            # Feed context place holders with input words  
            out = sess.run(output, feed_dict={context: [context_tokens]})


            print(str(round((time.time() - start_time)*1000, 1))+'ms')
            i=0
            #printing out the predictions.

            for word in out[0][0]:
            	wordarray = [ word ]
            	print(out[1][0][i])
            	print( raw_text+' ' + enc.decode(wordarray) )
            	i+=1
            print("")

    
    return print('done with predictions')
    



Test_and_save_simple_graph(input_test_file='gpt-3_test_input.json',Original=True)
















