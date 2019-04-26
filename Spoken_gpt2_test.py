import fire
import json
import os
import numpy as np
import tensorflow as tf



#!/usr/bin/env python3

import fire
import json
import time 
import os
import numpy as np
import tensorflow as tf
import json
import model, sample_spoken_edit, encoder

def interact_model(input_test_file=None,model_name='117M', seed=None, nsamples=1, batch_size=1, length=1, temperature=1, top_k=10, ):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    test_input=open(input_test_file)
    inputs_words = json.load(test_input)
    test_input.close()
    

    start_time = time.time()
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))



    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        

        output = sample_spoken_edit.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )
        
        tf.print(output)
        saver = tf.train.Saver()
        # chnage the name of the check point file if required 
        ckpt = tf.train.latest_checkpoint(os.path.join('checkpoint', 'run1'))
        saver.restore(sess, ckpt)
        print(round((time.time() - start_time)*1000, 2)+' time to intialise model in milli Sec')


        for raw_text in inputs_words:
            start_time = time.time()
            context_tokens = enc.encode(raw_text)
            out = sess.run(output, feed_dict={context: [context_tokens for _ in range(batch_size)]})
            print(round((time.time() - start_time)*1000, 2)+'milli Sec')

            for word in out[0]:
            	wordarray = [ word ]
            	print( raw_text+' ' + enc.decode(wordarray) )
            print("=" * 80 )
    
    return print('done with predictions')
    



interact_model(input_test_file='gpt-3_test_input.json', model_name='117M',seed=None,nsamples=1,batch_size=1,length=1,temperature=1,top_k=10)


# my intial plan is to run the model with in the for loop of inputs from json file, but if we do that each time we have to load the model and do single prediction

# dis advantage of this is ruturn statements 

# we should try nsamples=10 to know how it looks. i am not sure exact fucntionality top_k=10 is doing here, (it say Top_k is number of words considered at a time what if this is like considering )

#  try even  length =10 , as it says number of tokens returned by the model














