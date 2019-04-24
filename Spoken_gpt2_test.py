import fire
import json
import os
import numpy as np
import tensorflow as tf

from interactive_conditional_samples import interact_model


interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,batch_size=1,length=1,temperature=1,top_k=10)



