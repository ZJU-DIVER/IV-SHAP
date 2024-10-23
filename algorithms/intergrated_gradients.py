from tensorflow.keras.models import load_model
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
import keras
import random
from tqdm import trange
import time

def calculate_mlp_integrated_gradients(model_name, input_data, baseline, steps=50):
    model = load_model('./model_parameters/{}.h5'.format(model_name))

    #model.compile(optimizer='adam', loss='mse')
    # Calculate the difference between the two
    input_data = np.array(input_data).reshape(1,-1)
    baseline = np.array(baseline).reshape(1,-1)
    # Ensure the input is a TensorFlow tensor
    if not isinstance(input_data, tf.Tensor):
        input_data = tf.convert_to_tensor(input_data)
    if not isinstance(baseline, tf.Tensor):
        baseline = tf.convert_to_tensor(baseline)
    input_data = tf.cast(input_data, dtype=tf.float32)
    baseline = tf.cast(baseline, dtype=tf.float32)
    delta = input_data - baseline
    # Initialize integrated gradients
    integrated_gradients = tf.zeros_like(input_data)
    # Use TensorFlow's GradientTape for automatic differentiation
    start_time = time.time()
    for step in np.linspace(0, 1, steps):
        with tf.GradientTape() as tape:
            # Interpolate input
            interpolated_input = baseline + step * delta
            tape.watch(interpolated_input)
            # Run forward propagation of the model
            outputs = model(interpolated_input)
            # Calculate gradients relative to the input
            grads = tape.gradient(outputs, interpolated_input)

        # Update integrated gradients
        integrated_gradients += grads * (1.0 / steps) * delta

    end_time = time.time()  # End timing
    runtime = end_time - start_time  # Calculate runtime
    print(f"Runtime of IG: {runtime} seconds")
    return integrated_gradients.numpy()

def calculate_iv_integrated_gradients(model_name, input_data, baseline, steps=50):
    model = load_model('./model_parameters/{}.h5'.format(model_name))

    #model.compile(optimizer='adam', loss='mse')
    # Calculate the difference between the two
    t = np.array([input_data[0]])
    x = np.array([input_data[1:]])
    t_baseline = np.array([baseline[0]])
    x_baseline = np.array([baseline[1:]])

    # Ensure the input is a TensorFlow tensor
    if not isinstance(t, tf.Tensor):
        t = tf.convert_to_tensor(t, dtype=tf.float32)
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if not isinstance(t_baseline, tf.Tensor):
        t_baseline = tf.convert_to_tensor(t_baseline, dtype=tf.float32)
    if not isinstance(x_baseline, tf.Tensor):
        x_baseline = tf.convert_to_tensor(x_baseline, dtype=tf.float32)
    t_delta = t - t_baseline
    x_delta = x - x_baseline
    # Initialize integrated gradients
    integrated_gradients_t = tf.zeros_like(t)
    integrated_gradients_x = tf.zeros_like(x)
    # Use TensorFlow's GradientTape for automatic differentiation
    for step in np.linspace(0, 1, steps):
        with tf.GradientTape(persistent=True) as tape:
            # Interpolate input
            interpolated_t = t_baseline + step * t_delta
            interpolated_x = x_baseline + step * x_delta
            tape.watch(interpolated_t)
            tape.watch(interpolated_x)

            # Run forward propagation of the model
            outputs = model([interpolated_t, interpolated_x])
            # Calculate gradients relative to the input
            grads_t = tape.gradient(outputs, interpolated_t)
            grads_x = tape.gradient(outputs, interpolated_x)
            
        # Update integrated gradients
        integrated_gradients_t += grads_t * (1.0 / steps) * t_delta
        integrated_gradients_x += grads_x * (1.0 / steps) * x_delta
    return tf.concat([tf.expand_dims(integrated_gradients_t, axis=-1), integrated_gradients_x], axis=-1).numpy()

def calculate_mlp_integrated_gradients_unbiased(model_name, input_data, baseline, steps=50):
    model = load_model('./model_parameters/{}.h5'.format(model_name))

    #model.compile(optimizer='adam', loss='mse')
    # Calculate the difference between the two
    input_data = np.array(input_data).reshape(1,-1)
    baseline = np.array(baseline).reshape(1,-1)
    # Ensure the input is a TensorFlow tensor
    if not isinstance(input_data, tf.Tensor):
        input_data = tf.convert_to_tensor(input_data)
    if not isinstance(baseline, tf.Tensor):
        baseline = tf.convert_to_tensor(baseline)
    input_data = tf.cast(input_data, dtype=tf.float32)
    baseline = tf.cast(baseline, dtype=tf.float32)
    delta = input_data - baseline
    # Initialize integrated gradients
    integrated_gradients = tf.zeros_like(input_data)
    # Use TensorFlow's GradientTape for automatic differentiation
    for i in trange(steps):
        with tf.GradientTape() as tape:
            # Interpolate input
            step = random.random()
            interpolated_input = baseline + step * delta
            tape.watch(interpolated_input)
            # Run forward propagation of the model
            outputs = model(interpolated_input)
            # Calculate gradients relative to the input
            grads = tape.gradient(outputs, interpolated_input)

        # Update integrated gradients
        integrated_gradients += grads * (1.0 / steps) * delta
    return integrated_gradients.numpy()
