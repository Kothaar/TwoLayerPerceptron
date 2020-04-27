#!/usr/bin/env python3

import numpy as np
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
#test_images = gzip.open('test.idx3-ubyte', 'rb') 
#test_labels =gzip.open('test.idx3-ubyte', 'rb') 

    
## Magic Number tells us if its data or labels

# Training Images Magic Number
content = train_images.read(4)
magic_train_images = int.from_bytes(content,'big')
print("Training Images Magic Number?: ", magic_train_images)

# Training Labels Magic Number
content = train_labels.read(4)
magic_train_labels = int.from_bytes(content,'big')
print("Training Labels Magic Number?: ", magic_train_labels)

## Number of images/labels in file

# Number of Training Images
content = train_images.read(4)
number_images_training= int.from_bytes(content,'big')
print("Training Number of Images: ", number_images_training)
# Number of Training Lables
content = train_labels.read(4)
number_labels_training = int.from_bytes(content,'big')
print("Training Number of Labels: ", number_labels_training)

## Number of Rows and Columns

# Numbers of rows per image
content = train_images.read(4)
rows_train = int.from_bytes(content,'big')
print("Training Rows: ", rows_train)

# Numbers of columns per image
content = train_images.read(4)
columns_train = int.from_bytes(content,'big')
print("Training Columns: ", columns_train)

################################################################################
### Filling Training Image Matrix
################################################################################

# Calculate remaining # of bits in file
remaining_bytes = number_images_training * rows_train * columns_train

# struck.unpack doc https://docs.python.org/2/library/struct.html
# Creates a 2-D matrix each layer containing an image
train_array_images = np.array(struct.unpack('>' + 'B' * remaining_bytes,
    train_images.read(remaining_bytes))).reshape((number_images_training,
        rows_train * columns_train))

print("Loading Input Array")
print("Array Dimensions: ", train_array_images.shape)
print("Should match: ", "(", number_images_training, ",", columns_train * rows_train, ")")

# Scaling each feature to a fracton between 0 and 1 to avoid very large weights
train_array_images = train_array_images / 255.0
print("Scaling array to be between 0 and 1")
print(train_array_images[1],"\n\n")

### Filling Training Lables Matrix
# Creates a 1-D matrix of labels
train_array_labels = np.array(struct.unpack('>' + 'B' * number_labels_training,
    train_labels.read(number_labels_training)))

# weight array with 785 random numbers between -0.5 and 0.5
print("Randomizing weight matrix")
weight_array = np.random.uniform(-.05,0.05,size=(785,10))

# Create array of 1's for bias
bias_stack = np.full(shape= number_images_training, fill_value=1, dtype=np.float)

# 1d array -> 2d array to allow concatonation
bias_stack = np.array([bias_stack])

# Add bias to front of training image array
training_array = np.concatenate((bias_stack.T, train_array_images), axis=1)
print("Appending bias stack to input matrix")


# How many times train data
EPOCH = 70 
trainingRate = 0.01

# Epochs Accuracy
acc = 0

confusion_matrix = np.zeros([10,10])
accuracy = np.empty(0)

#############################################################
# EPOCH 0
#############################################################

print("TESTING EPOCH 0 - No weight adjustments made")
# Iterate through each Image and it's label together
for image, target in zip(training_array, train_array_labels):

    # Calculate which Neurons fires per image
    activation = np.dot(image.T, weight_array)

    # Guess is set to the index which contains the largest element
    guess  = np.argmax(activation) 

    # Checks to see if the index matches the target value
    if target == guess:
        acc += 1

    # Convert activation vector to 1's and 0's baed on if it fired (ie > 0)
    activation = np.where(activation > 0, 1,0)
    
    # Increment value in confusion matrix
    confusion_matrix[guess][target] += 1
    
print("Total Correct Guesses: ", acc)
print("Accuracy: ", acc/number_images_training)
accuracy = np.append(accuracy, acc/number_images_training)

############################################################
# TRAINING
############################################################
print("\n\n Beginning Training for", EPOCH, "epochs\n\n")
# until iterated x times or all outputs correct
for epochs in range(EPOCH):

    print("TRAINING EPOCH", epochs+1)
    acc = 0

    # for each input row
    for image, target in zip(training_array, train_array_labels):

        # Calculate which Neurons fires per image
        activation = np.dot(image, weight_array)

        # Guess is set to the index which contains the largest element
        guess  = np.argmax(activation) 
        
        # Checks to see if the index matches the target value
        if guess == target:
            acc += 1

        # Convert activation vector to 1's and 0's baed on if it fired (ie > 0)
        activation = np.where(activation > 0, 1,0)

        # Count used to track which Neuron is being trained
        count = 0
        

        # For Each Neuron (Passes in the Column for each weight as a row)
        for neuron_weights, fired in zip(weight_array.T, activation):
            
            # If looking at the targets weights
            if count == target:
                current = 1
            else: 
                current = 0
            
            # Calculate w + deltaw
            neuron_weights += trainingRate*np.dot(image, current - fired)
            
            # Update weights
            weight_array.T[count] = neuron_weights
            
            # Increment to update the next Weight row
            count += 1

        # Increment position in confusion matrix
        confusion_matrix[guess][target] += 1

    print("Total Correct Gueses: ", acc)
    print("Accuracy: ", acc/number_images_training)
    accuracy = np.append(accuracy, acc/number_images_training)
    

############################################################
# Training Resaults
############################################################

# Change Accuracy from decimal to percentage
accuracy = accuracy * 100

confusion_matrix = confusion_matrix  
print(confusion_matrix)


############################################################
# Training Confusion Matrix
# source Used https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
############################################################



print(accuracy)

plt.plot(accuracy)
plt.title(trainingRate)
plt.xlabel('Epoch')
plt.ylim(0,100)
plt.xlim(0,int(EPOCH))
plt.ylabel('Accuracy')
plt.savefig(str(trainingRate)+'_linegraphtest.png')

    

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

plt.savefig(str(trainingRate)+'_ConfusionMatrixtest.png')
















