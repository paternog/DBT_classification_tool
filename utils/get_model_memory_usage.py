##########################################################################
# Author: Gianfranco Paternò (paterno@fe.infn.it)                        #
# collaboration of INFN Ferrara and INFN Napoli for the next_AIM project #
# Last modification date: March 2024                                     #
#                                                                        #
# SPDX-License-Identifier: EUPL-1.2                                      #
# Copyright 2024 Istituto Nazionale di Fisica Nucleare                   #
##########################################################################


def get_model_memory_usage(batch_size, model):
    """
    Return the estimated memory usage of a given Keras model in GB.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
        model: A Keras model.
    Returns:
        An estimate of the Keras model's memory usage in GB.
    
    gpaterno (10/01/2025):
        I made various modifications to make it work with tensorflow > 2.14.
    """
    import numpy as np
    import tensorflow as tf
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += get_model_memory_usage(batch_size, layer)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output.shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if not out_shape == None:
            for s in out_shape:
                if s is None:
                    continue
                single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )
    
    total_memory = (
        batch_size * shapes_mem_count
        + trainable_count
        + non_trainable_count
    )

    total_memory = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    
    return total_memory
