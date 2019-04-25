import fire
import json
import os
import numpy as np
import tensorflow as tf

from interactive_conditional_samples import interact_model


#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import json
import model, sample, encoder

def interact_model(model_name='117M', seed=None, nsamples=1, batch_size=1, length=1, temperature=1, top_k=10, input_test_file):
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
    

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
## ???   we have to get the right encoder and h params. json in the model    (encode.json is a word_id dictionary )
#?? vocab.bpe is the vocab_list thats being used inside encoder we should provide these files in order to test, if they are not saved automatically. 
### ??? simplest solutions is to test it with what we got for now ## we can just over write name of the model_name at a place 
# word_to_id that we have cannot be used beacuse we have cleaned that data with a different dictionary, so now we may need to look at encode.py dictioanry 
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = 1

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        

        #np.random.seed(seed)  #  lets see if we can run without random seed 
        #tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        # chnage the name of the check point file if required 
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        start_time = time.time()


        for raw_text in inputs_words:
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={context: [context_tokens for _ in range(batch_size)]})[:, len(context_tokens):]

                for i in range(batch_size):
                    # since batch_size is one generated = text; but lets see how it works 
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
                print((time.time() - start_time))

    return print('done with predictions')
    



interact_model(model_name='117M',seed=None,nsamples=1,batch_size=1,length=1,temperature=1,top_k=10,input_test_file='gpt-3_test_input.json')


# my intial plan is to run the model with in the for loop of inputs from json file, but if we do that each time we have to load the model and do single prediction

# dis advantage of this is ruturn statements 

# we should try nsamples=10 to know how it looks. i am not sure exact fucntionality top_k=10 is doing here, (it say Top_k is number of words considered at a time what if this is like considering )

#  try even  length =10 , as it says number of tokens returned by the model


























interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,batch_size=1,length=1,temperature=1,top_k=10)



