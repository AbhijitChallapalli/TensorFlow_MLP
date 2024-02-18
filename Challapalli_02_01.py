import numpy as np
import tensorflow as tf


def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):

    # Splitting the data into train and validation 
    # Reference from the helpers.py file given by professor
    def split_data(X_train, Y_train, split_range=[0.2, 0.7]):
        start = int(split_range[0] * X_train.shape[0])
        end = int(split_range[1] * X_train.shape[0])
        return np.concatenate((X_train[:start], X_train[end:])), np.concatenate((Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]

    # Creating train and val data
    # Reference from the helpers.py file given by professor
    X_train, Y_train, X_val, Y_val = split_data(X_train,Y_train, split_range=validation_split)

    # Generating the batches
    # Reference from the helpers.py file given by professor
    def generate_batches(X, y, batch_size=32):
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
        if X.shape[0] % batch_size != 0:
            yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]


    #If the weighs passed into the network use those otherwise i.e. if weights are none initialize weights 
    if weights is None:
        weights= []
        a=[X_train.shape[1]]
        layers =  a + layers
        for i, layer in enumerate(layers): 
            if i < (len(layers)-1):
                np.random.seed(seed)
                w = tf.Variable(np.random.randn(layer+1,layers[i+1]),dtype=tf.float32)
                weights.append(w)
    else:
        weights = weights
    error_each_epoch = []


    # Calculating the net value and applying activation functions as specified
    def calculate_net(X, weights, activations):   
        for weight, activation in zip(weights, activations):
            net = tf.matmul(X, weight[1:, :]) + weight[0]
            if activation == 'relu':
                X = tf.nn.relu(net)
            elif activation == 'sigmoid':
                X = tf.nn.sigmoid(net)
            elif activation == 'linear':
                X = net
            else:
                raise ValueError("Provide activation functions as any of the following : relu linear sigmoid")
        return X

    #Defining the loss functions svm , mse , cross_entropy
    #svm max(0,y-yst+delta)
    # mse square(y-yhat)
    # cross entropy is calculated using the tf.nn.softmax_cross_entropy_with_logits
    def calculate_loss(Y, Y_pred, loss):
        if loss.lower() == "svm":
            calculated_loss=tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0,Y - Y_pred + 5.)))
        elif loss.lower() == "mse":
            calculated_loss = tf.reduce_mean(tf.square(Y - Y_pred))
        elif loss.lower() == "cross_entropy":
            calculated_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
        else:
            raise ValueError("Loss function should be either of svm mse cross_entropy ", loss)

        return calculated_loss


    #Without modifying the weights i.e. freezing the weights
    if epochs==0:
        actual_output = []
        cal_loss = []
        for x_batch, y_batch in generate_batches(X_val, Y_val, batch_size=1):
            net = calculate_net(x_batch, weights, activations)
            actual_output.extend(net)
            val_loss = calculate_loss(y_batch, net, loss)
            cal_loss.append(val_loss)
        error_each_epoch.append(np.mean(cal_loss))
        return weights, error_each_epoch, np.array(actual_output)

    # Training neural network
    for epoch in range(epochs):
        cal_loss = []
        actual_output = []
        for x_batch, y_batch in generate_batches(X_train, Y_train, batch_size):
            with tf.GradientTape() as tape:
                tape.watch(weights)
                train_prediction = calculate_net(x_batch, weights, activations)
                train_loss = calculate_loss(y_batch, train_prediction, loss)
            # Calculating the gradients
            gradient = tape.gradient(train_loss, weights)      
            # Updating the weights
            for i in range(len(weights)):
                weights[i] = weights[i] - alpha*gradient[i]

        # Validation Dataset
        for x_batch, y_batch in generate_batches(X_val, Y_val, batch_size=1):
            net = calculate_net(x_batch, weights, activations)
            actual_output.extend(net)
            val_loss = calculate_loss(y_batch, net, loss)
            cal_loss.append(val_loss)

        error_each_epoch.append(np.mean(cal_loss))

    return weights, error_each_epoch, np.array(actual_output)
