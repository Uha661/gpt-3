import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def sample_sequence(*, hparams, length, start_token=None, context=None, temperature=1, top_k=10):

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
            
        context_output = step(hparams, context[:, :-1])

        # initializing tensors of shape (1,top_k) to copy top_10 predictions
        top_10 = tf.zeros(shape=[1,top_k],dtype=tf.dtypes.int32,name=None)
        top_10_probablities = tf.zeros(shape=[1,top_k],dtype=tf.dtypes.float32,name=None)


        def body(past, prev, output,top_10,top_10_probablities):

            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
           

            # acessing top_10 predictions from logits
            top_10_probablities, top_10 = tf.nn.top_k(logits,k=top_k,sorted=True,name='probablities')
            

            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            

            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                top_10,
                top_10_probablities,
            ]

        def cond(*args):
            return True



            # we have added top_10 and top_10_probablities as the loop_vars and added shape_invariants to the loop. 
        _, _,_,tokens,token_probablities = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                top_10,
                top_10_probablities,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams)),
                tf.TensorShape([1]),
                tf.TensorShape([1, None]),
                tf.TensorShape([1,top_k]),
                tf.TensorShape([1,top_k]),

            ],
            back_prop=False,
        )
        
        return tokens,token_probablities
