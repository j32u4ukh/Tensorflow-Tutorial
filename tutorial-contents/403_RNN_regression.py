"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
TIME_STEP = 10       # rnn time step
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)

# float32 for converting torch FloatTensor
x_np = np.sin(steps)
y_np = np.cos(steps)

plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])  # shape(batch, 5, 1)
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])  # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)

# very first hidden state
init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
outputs, final_s = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)

# reshape 3D output to 2D for fully connected layer
outs2D = tf.reshape(outputs, [-1, CELL_SIZE])
net_outs2D = tf.layers.dense(outs2D, INPUT_SIZE)

# reshape back to 3D
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])

# compute cost
loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()

# initialize var in graph
sess.run(tf.global_variables_initializer())

plt.figure(1, figsize=(12, 5))

# continuously plot
plt.ion()

for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
    y = np.cos(steps)[np.newaxis, :, np.newaxis]

    if 'final_s_' not in globals():                 # first state, no any hidden state
        feed_dict = {tf_x: x, tf_y: y}
    else:                                           # has hidden state, so pass it to rnn
        feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}

    # train
    _, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)

    # plotting
    plt.plot(steps, y.flatten(), 'r-')
    plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
sess.close()
