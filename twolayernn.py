#!/usr/bin/env python3
# Author: Kelly Burton
# Used for Experiment 1 of Programming assignment 1

import numpy as np
import math
import gzip
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




################################################################################
# LOADING DATA
################################################################################

train_images = gzip.open('train_images.gz', 'rb') 
train_labels = gzip.open('train_labels.gz', 'rb') 
test_images = gzip.open('test_images.gz', 'rb') 
test_labels =gzip.open('test_labels.gz', 'rb') 

    
## Magic Number tells us if its data or labels

# Training Images Magic Number
content = train_images.read(4)
magic_train_images = int.from_bytes(content,'big')
print("Training Images Magic Number?: ", magic_train_images)

# Training Labels Magic Number
content = train_labels.read(4)
magic_train_labels = int.from_bytes(content,'big')
print("Training Labels Magic Number?: ", magic_train_labels)

# Training Labels Magic Number
content = test_images.read(4)
magic_test_images = int.from_bytes(content,'big')
print("Training Labels Magic Number?: ", magic_test_images)

# Training Labels Magic Number
content = test_labels.read(4)
magic_test_labels = int.from_bytes(content,'big')
print("Training Labels Magic Number?: ", magic_test_labels)

## Number of images/labels in file

# Number of Training Images
content = train_images.read(4)
number_images_training= int.from_bytes(content,'big')
print("Training Number of Images: ", number_images_training)
# Number of Training Lables
content = train_labels.read(4)
number_labels_training = int.from_bytes(content,'big')
print("Training Number of Labels: ", number_labels_training)
# Number of Training Images
content = test_images.read(4)
number_images_test= int.from_bytes(content,'big')
print("Training Number of Images: ", number_images_test)
# Number of Training Lables
content = test_labels.read(4)
number_labels_test = int.from_bytes(content,'big')
print("Training Number of Labels: ", number_labels_test)

## Number of Rows and Columns

# Numbers of rows per image
content = train_images.read(4)
rows_train = int.from_bytes(content,'big')
print("Training Rows: ", rows_train)
# Numbers of rows per image
content = test_images.read(4)
rows_test= int.from_bytes(content,'big')
print("Training Rows: ", rows_test)

# Numbers of columns per image
content = train_images.read(4)
columns_train = int.from_bytes(content,'big')
print("Training Columns: ", columns_train)
# Numbers of columns per image
content = test_images.read(4)
columns_test = int.from_bytes(content,'big')
print("Training Columns: ", columns_test)

#############################################################
# Parameters 
#############################################################


# How many times train data
EPOCH = 50
# training rate aka ETA
trainingRate = 0.1
momentum = 0.9
# size of hidden layer
hidden_nodes = 20

################################################################################
### Functions
################################################################################

def sigmoid(trainingRate, activation):
    sig = lambda x: (1 / (1 + math.e**(-1 * trainingRate * x)))
    activation = sig(activation)
    return activation

################################################################################
### Filling Training Matraacies
################################################################################

print("Loading Training Arrays")
# Calculate remaining # of bits in file
remaining_bytes_train = number_images_training * rows_train * columns_train
remaining_bytes_test = number_images_test * rows_test * columns_test

# struck.unpack doc https://docs.python.org/2/library/struct.html
# Creates a 2-D matrix each layer containing an image
# train array
train_array_images = np.array(struct.unpack('>' + 'B' * remaining_bytes_train,
    train_images.read(remaining_bytes_train))).reshape((number_images_training,
        rows_train * columns_train))

# Test array
test_array_images = np.array(struct.unpack('>' + 'B' * remaining_bytes_test,
    test_images.read(remaining_bytes_test))).reshape((number_images_test,
        rows_test * columns_test))


# Scaling each feature to a fracton between 0 and 1 to avoid very large weights
train_array_images = train_array_images / 255.0
test_array_images = test_array_images / 255.0

### Filling Training Lables Matrix
# Creates a 1-D matrix of labels
# Training Labels
train_array_labels = np.array(struct.unpack('>' + 'B' * number_labels_training,
    train_labels.read(number_labels_training)))

# Testing labels
test_array_labels = np.array(struct.unpack('>' + 'B' * number_labels_test,
    test_labels.read(number_labels_test)))


#############################################################
#  Experiment 1
# Training with n set to 20, 50, and 100
#############################################################


n = hidden_nodes
print("\n\n\nExperiment 1")
print("Testing with n = ", n ,"\n\n\n")

confusion_matrix = np.zeros([10,10])
train_accuracy = np.empty(0)
test_accuracy = np.empty(0)

    

# Initialize weight arrays with random number from -.05 to 0.5.
print("Randomizing first weight matrix")
weight_array_1 = np.random.uniform(-.05,0.05,size=(785,hidden_nodes))
print("Randomizing second weight matrix")
weight_array_2 = np.random.uniform(-.05,0.05,size=(hidden_nodes+1,10))


## Adding Bias to Matricies

# Create array of 1's for bias
bias_stack = np.full(shape= number_images_training, fill_value=1, dtype=np.float)
# 1d array -> 2d array to allow concatonation
bias_stack = np.array([bias_stack])
# Add bias to front of training image array
training_array = np.concatenate((bias_stack.T, train_array_images), axis=1)

# Create array of 1's for bias
bias_stack = np.full(shape= number_images_test, fill_value=1, dtype=np.float)
# 1d array -> 2d array to allow concatonation
bias_stack = np.array([bias_stack])
# Add bias to front of training image array
testing_array = np.concatenate((bias_stack.T, test_array_images), axis=1)
print("Appending bias stack to input matrix")



# Create hidden layer Node Matrix
hidden_array = np.array(hidden_nodes)

# Initalizing delta weight arrays
delta_w1 = np.copy(weight_array_1)
delta_w1.fill(0)
delta_w2 = np.copy(weight_array_2)
delta_w2.fill(0)

#############################################################
# EPOCH 0 Training set
#############################################################
acc_train = 0

print("\n\nTraining EPOCH 0 - No weight adjustments made")
# Iterate through each Image and it's label together
for image, target in zip(training_array, train_array_labels):


    # Calculate which Neurons fires per image
    a = np.array([1])
    a = np.append(a, np.dot(image.T, weight_array_1))

    # Using sigmoid function on activation vector
    a = sigmoid(trainingRate, a)
    
    y = np.dot(a, weight_array_2)
    y = sigmoid(trainingRate, y)



    guess  = np.argmax(y) 

    # Checks to see if the index matches the target value
    if target == guess:
        acc_train += 1

    # Increment value in confusion matrix
    
print("Accuracy: ", acc_train/number_images_training)
train_accuracy = np.append(train_accuracy, acc_train/number_images_training)

#############################################################
# EPOCH 0 Testing set
#############################################################
acc_test = 0

print("\n\nTESTING EPOCH 0 - No weight adjustments made")
# Iterate through each Image and it's label together
for image, target in zip(testing_array, test_array_labels):


    # Calculate which Neurons fires per image
    a = np.array([1])
    a = np.append(a, np.dot(image.T, weight_array_1))

    # Using sigmoid function on activation vector
    a = sigmoid(trainingRate, a)
    
    y = np.dot(a, weight_array_2)
    y = sigmoid(trainingRate, y)



    guess  = np.argmax(y) 

    # Checks to see if the index matches the target value
    if target == guess:
        acc_test += 1

    
print("Accuracy: ", acc_test/number_images_test)
test_accuracy = np.append(test_accuracy, acc_test/number_images_test)


############################################################
# TRAINING
############################################################
print("\n\n Beginning Training for", EPOCH, "epochs\n\n")
# until iterated x times or all outputs correct
for epochs in range(EPOCH):

    print("TRAINING EPOCH", epochs+1)
    acc_train = 0

    delta_w1.fill(0)
    delta_w2.fill(0)
    shuffle = np.arange(training_array.shape[0])
    np.random.shuffle(shuffle)


    # for each input row
    for image, target in zip(training_array[shuffle], train_array_labels[shuffle]):

        ############################################################
        # Forward Propigation
        ############################################################

        # Forward Propagate input to hidden array w/ bias attached in 0 index
        a = np.array([1])
        a = np.append(a, np.dot(image.T, weight_array_1))
        # Applying Activation Function
        a = sigmoid(trainingRate, a)
        
        # Forward Propagate hidden array to output
        y = np.dot(a, weight_array_2)
        # Applying Activation Function
        y = sigmoid(trainingRate, y)

        # Interpret the output layer as a classification
        guess  = np.argmax(y) 

        # Activation output changed to .9 and .1 if fired or not
        activ = np.where(y > 0, .9, .1)

        if target == guess:
            acc_train += 1



        # setting t_k to .9 if it is the target otherise .01
        t = np.ones(10)/10
        t[target] = .9


        ############################################################
        # Backword Propigation
        ############################################################


        # Calculate error for output and hidden layer
        #error_o = (y - t) * y * (1 - y)
        error_o = y * (1 - y) * (t - y)
        error_h = a * (1 - a) * np.dot(weight_array_2 , error_o)

        # Reshape arrays to allow multiplication and then finding the changing for hidden weights
        error2 = np.reshape(a, (1,a.shape[0]))
        hidden_activation = np.reshape(error_o, (error_o.shape[0],1))
        delta_w2 = trainingRate * np.outer(error2, hidden_activation)  + momentum * delta_w2
        weight_array_2 += delta_w2


        # Reshape arrays to allow multiplication and then finding the changing for input weights
        error1 = np.reshape(error_h, (1,error_h.shape[0]))
        input_data = np.reshape(image, (image.shape[0]))

        # Remove Bias at the front of vector
        error1 = np.delete(error1, 0,1)

        delta_w1 = trainingRate * np.outer(error1 , input_data).T  + momentum * delta_w1

        weight_array_1 += delta_w1


        # Increment position in confusion matrix

    print("Traing Accuracy: ", acc_train/number_images_training)
    train_accuracy = np.append(train_accuracy, acc_train/number_images_training)
        ############################################################
        # Testing 
        ############################################################
    acc_test = 0
    for image, target in zip(testing_array, test_array_labels):

        # Forward Propagate input to hidden array w/ bias attached in 0 index
        a = np.array([1])
        a = np.append(a, np.dot(image.T, weight_array_1))
        # Applying Activation Function
        a = sigmoid(trainingRate, a)
        
        # Forward Propagate hidden array to output
        y = np.dot(a, weight_array_2)
        # Applying Activation Function
        y = sigmoid(trainingRate, y)

        # Interpret the output layer as a classification
        guess  = np.argmax(y) 

        # Activation output changed to .9 and .1 if fired or not
        activ = np.where(y > 0, .9, .1)

        if target == guess:
            acc_test += 1


    print("Test Accuracy: ", acc_test/number_images_test)
    test_accuracy = np.append(test_accuracy, acc_test/number_images_test)
############################################################
# Training End
############################################################
############################################################
# Confusion Matrix 
############################################################

print("Creating Confusion Matrix based on trained data")

acc_test = 0
for image, target in zip(testing_array, test_array_labels):

    # Forward Propagate input to hidden array w/ bias attached in 0 index
    a = np.array([1])
    a = np.append(a, np.dot(image.T, weight_array_1))
    # Applying Activation Function
    a = sigmoid(trainingRate, a)
    
    # Forward Propagate hidden array to output
    y = np.dot(a, weight_array_2)
    # Applying Activation Function
    y = sigmoid(trainingRate, y)

    # Interpret the output layer as a classification
    guess  = np.argmax(y) 

    # Activation output changed to .9 and .1 if fired or not
    activ = np.where(y > 0, .9, .1)

    confusion_matrix[guess][target] += 1



############################################################
# Training Resaults
############################################################

# Change Accuracy from decimal to percentage
train_accuracy = train_accuracy * 100
test_accuracy = test_accuracy * 100

############################################################
# Training Confusion Matrix
# source Used https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
############################################################
plt.figure()

fig, ax = plt.subplots(figsize=(20,20))
im = ax.imshow(confusion_matrix)
ax.xaxis.tick_top()

for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
        text = ax.text(i,j,int(confusion_matrix[i,j]), ha="center",va="center", color ="w")

fig.tight_layout()
ax.set_title("Confusion Matrix")
plt.xlabel('Actual')
plt.ylabel('Guess')

plt.savefig(str(hidden_nodes)+'_e1_conf.png')



plt.figure()
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['Training Accuracy', 'Testing Accuracy'])
plt.title(hidden_nodes)
plt.xlabel('Epoch')
plt.ylim(0,100)
plt.xlim(0,int(EPOCH))
plt.ylabel('Accuracy')
plt.savefig(str(hidden_nodes)+'_e1_line.png')





