# # Hello sir

# # I would like to mention a few glitches I faced when I was implementing the file. I would like to mention them all to ensure that the application runs effectively in case you want to test it otherwise. 

# # --> Apart from the packages you had mentioned in the tutorial I had to install package which was pydot to render the graph. 
# # --> I had to set the environment variable 'KMP_DUPLICATE_LIB_OK']='True' to avoid error while running it 
# # --> I have preferred CNNs over MLPs because of their tolerance to position 
# # --> To improve generalization and tolerance to new values, drop out regularization has been used
# # --> Further, more types of regulaization which has not been mentioned in the tutorial such as kernel, bias and activity regularization have also been used.
# # --> To make sure the model is tolerant in size, phase and angle variations, Data augmentation using the Cut mix augmentation type has been implemented for preprocessing the data
# # Upon running for the test use cases, the percentage correctly classified was about 98.63 of the test data

# # Importing files 

# import numpy as np
# import keras
# import tensorflow
# import os 
# import timeit 
# import matplotlib.pyplot as plt 
# from tensorflow.keras.datasets import mnist
# from keras import layers
# from tensorflow.keras.models import load_model
# from tensorflow import clip_by_value
# from tensorflow import data as tf_data
# from tensorflow import image as tf_image
# from tensorflow import random as tf_random
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten,Dropout, Input
# from keras import regularizers
# from tensorflow.keras.utils import plot_model

# # Timing the entire exceution time 

# start = timeit.timeit()
# print("Started the timing")

# # Setting the envirnoment variables

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# keras.utils.set_random_seed(42)

# # Loading the data

# (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
# y_train = keras.utils.to_categorical(labels_train, num_classes=10)
# y_test = keras.utils.to_categorical(labels_test, num_classes=10)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # Data Augmentation

# AUTO = tf_data.AUTOTUNE
# BATCH_SIZE = 1
# IMG_SIZE = 28

# train_ds_one = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))
# train_ds_two = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))

# train_ds_simple = tf_data.Dataset.from_tensor_slices((x_train, y_train))
# test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test))

# train_ds_simple = (train_ds_simple.batch(BATCH_SIZE).prefetch(AUTO))
# train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

# def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
#     gamma_1_sample = tf_random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf_random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# def get_box(lambda_value):
#     cut_rat = tensorflow.math.sqrt(1.0 - lambda_value)

#     cut_w = IMG_SIZE * cut_rat 
#     cut_w = tensorflow.cast(cut_w, "int32")

#     cut_h = IMG_SIZE * cut_rat
#     cut_h = tensorflow.cast(cut_h, "int32")

#     cut_x = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # rx
#     cut_x = tensorflow.cast(cut_x, "int32")
#     cut_y = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # ry
#     cut_y = tensorflow.cast(cut_y, "int32")

#     boundaryx1 = clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
#     boundaryy1 = clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
#     bbx2 = clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
#     bby2 = clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

#     target_h = bby2 - boundaryy1
#     if target_h == 0:
#         target_h += 1

#     target_w = bbx2 - boundaryx1
#     if target_w == 0:
#         target_w += 1

#     return boundaryx1, boundaryy1, target_h, target_w


# def cutmix(train_ds_one, train_ds_two):
#     (image1, label1), (image2, label2) = train_ds_one, train_ds_two

#     alpha = [0.25]
#     beta = [0.25]

#     lambda_value = sample_beta_distribution(1, alpha, beta)

#     lambda_value = lambda_value[0][0]
#     boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

#     crop2 = tf_image.crop_to_bounding_box(image2, boundaryy1, boundaryx1, target_h, target_w)
#     image2 = tf_image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

#     crop1 = tf_image.crop_to_bounding_box(image1, boundaryy1, boundaryx1, target_h, target_w)


#     img1 = tf_image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
#     image1 = image1 - img1
#     image = image1 + image2

#     lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
#     lambda_value = tensorflow.cast(lambda_value, "float32")

#     label = lambda_value * label1 + (1 - lambda_value) * label2
#     return image, label

# train_ds_cmu = (train_ds.shuffle(1024).map(cutmix, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# train_ds_cmu=list(train_ds_cmu)
# test_ds=list(test_ds)
# x_train_revised=[]
# y_train_revised=[]
# x_test_revised=[]
# y_test_revised=[]
# x_test_revised=[]
# y_test_revised=[]

# for i in range (60000):
#     x_train_revised.append(train_ds_cmu[i][0][0])
#     y_train_revised.append(train_ds_cmu[i][1][0])

# for i in range (10000):
#     x_test_revised.append(test_ds[i][0])
#     y_test_revised.append(test_ds[i][1])

# x_train_revised=np.array(x_train_revised)
# y_train_revised=np.array(y_train_revised)
# x_test_revised=np.array(x_test_revised)
# y_test_revised=np.array(y_test_revised)

# # Defining the CNN model with regularization

# inputs = Input(shape=x_train.shape[1:])
# x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(5,5), activation='tanh',bias_regularizer=regularizers.L2(1e-4))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(5,5), activation='silu',activity_regularizer=regularizers.L2(1e-5))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid')(x)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(256, activation='softplus')(x)
# x = Dropout(rate=0.5)(x)
# outputs = Dense(10, activation='softmax')(x)
# net = Model(inputs=inputs, outputs=outputs)

# print(net.summary())
# plot_model(net, to_file='network_structure.png', show_shapes=True)
# net.compile(loss='categorical_crossentropy', optimizer='adam')

# # Fitting the model 

# history = net.fit(x_train_revised,y_train_revised,validation_data=(x_test_revised,y_test_revised),epochs=50,batch_size=250)

# # Plotting the error

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# net.save("Final_1.h5")
# net=load_model("Final_1.h5")

# outputs=net.predict(x_test)
# labels_predicted=np.argmax(outputs, axis=1)
# misclassified=sum(labels_predicted!=labels_test)
# print('Percentage misclassified = ',100*misclassified/labels_test.size)

# plt.figure(figsize=(8, 2))
# for i in range(0,8):
#     ax=plt.subplot(2,8,i+1)
#     plt.imshow(x_test[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r'))
#     plt.title(labels_test[i])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# for i in range(0,8):
#     output = net.predict(x_test[i,:].reshape(1, 28,28,1)) 
#     output=output[0,0:]
#     plt.subplot(2,8,8+i+1)
#     plt.bar(np.arange(10.),output)
#     plt.title(np.argmax(output))
#     plt.show()

# end = timeit.timeit()
# print("The time elapsed is ", end - start)

# # Hello sir

# # I would like to mention a few glitches I faced when I was implementing the file. I would like to mention them all to ensure that the application runs effectively in case you want to test it otherwise. 

# # --> Apart from the packages you had mentioned in the tutorial I had to install package which was pydot to render the graph. 
# # --> I had to set the environment variable 'KMP_DUPLICATE_LIB_OK']='True' to avoid error while running it 
# # --> I have preferred CNNs over MLPs because of their tolerance to position 
# # --> To improve generalization and tolerance to new values, drop out regularization has been used
# # --> Further, more types of regulaization which has not been mentioned in the tutorial such as kernel, bias and activity regularization have also been used.
# # --> To make sure the model is tolerant in size, phase and angle variations, Data augmentation using the Cut mix augmentation type has been implemented for preprocessing the data
# # Upon running for the test use cases, the percentage correctly classified was about 98.63 of the test data

# # Importing files 

# import numpy as np
# import keras
# import tensorflow
# import os 
# import timeit 
# import matplotlib.pyplot as plt 
# from tensorflow.keras.datasets import mnist
# from keras import layers
# from tensorflow.keras.models import load_model
# from tensorflow import clip_by_value
# from tensorflow import data as tf_data
# from tensorflow import image as tf_image
# from tensorflow import random as tf_random
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten,Dropout, Input
# from keras import regularizers
# from tensorflow.keras.utils import plot_model

# # Timing the entire exceution time 

# start = timeit.timeit()
# print("Started the timing")

# # Setting the envirnoment variables

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# keras.utils.set_random_seed(42)

# # Loading the data

# (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
# y_train = keras.utils.to_categorical(labels_train, num_classes=10)
# y_test = keras.utils.to_categorical(labels_test, num_classes=10)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # Data Augmentation

# AUTO = tf_data.AUTOTUNE
# BATCH_SIZE = 1
# IMG_SIZE = 28

# train_ds_one = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))
# train_ds_two = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))

# train_ds_simple = tf_data.Dataset.from_tensor_slices((x_train, y_train))
# test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test))

# train_ds_simple = (train_ds_simple.batch(BATCH_SIZE).prefetch(AUTO))
# train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

# def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
#     gamma_1_sample = tf_random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf_random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# def get_box(lambda_value):
#     cut_rat = tensorflow.math.sqrt(1.0 - lambda_value)

#     cut_w = IMG_SIZE * cut_rat 
#     cut_w = tensorflow.cast(cut_w, "int32")

#     cut_h = IMG_SIZE * cut_rat
#     cut_h = tensorflow.cast(cut_h, "int32")

#     cut_x = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # rx
#     cut_x = tensorflow.cast(cut_x, "int32")
#     cut_y = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # ry
#     cut_y = tensorflow.cast(cut_y, "int32")

#     boundaryx1 = clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
#     boundaryy1 = clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
#     bbx2 = clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
#     bby2 = clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

#     target_h = bby2 - boundaryy1
#     if target_h == 0:
#         target_h += 1

#     target_w = bbx2 - boundaryx1
#     if target_w == 0:
#         target_w += 1

#     return boundaryx1, boundaryy1, target_h, target_w


# def cutmix(train_ds_one, train_ds_two):
#     (image1, label1), (image2, label2) = train_ds_one, train_ds_two

#     alpha = [0.25]
#     beta = [0.25]

#     lambda_value = sample_beta_distribution(1, alpha, beta)

#     lambda_value = lambda_value[0][0]
#     boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

#     crop2 = tf_image.crop_to_bounding_box(image2, boundaryy1, boundaryx1, target_h, target_w)
#     image2 = tf_image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

#     crop1 = tf_image.crop_to_bounding_box(image1, boundaryy1, boundaryx1, target_h, target_w)


#     img1 = tf_image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
#     image1 = image1 - img1
#     image = image1 + image2

#     lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
#     lambda_value = tensorflow.cast(lambda_value, "float32")

#     label = lambda_value * label1 + (1 - lambda_value) * label2
#     return image, label

# train_ds_cmu = (train_ds.shuffle(1024).map(cutmix, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# train_ds_cmu=list(train_ds_cmu)
# test_ds=list(test_ds)
# x_train_revised=[]
# y_train_revised=[]
# x_test_revised=[]
# y_test_revised=[]
# x_test_revised=[]
# y_test_revised=[]

# for i in range (60000):
#     x_train_revised.append(train_ds_cmu[i][0][0])
#     y_train_revised.append(train_ds_cmu[i][1][0])

# for i in range (10000):
#     x_test_revised.append(test_ds[i][0])
#     y_test_revised.append(test_ds[i][1])

# x_train_revised=np.array(x_train_revised)
# y_train_revised=np.array(y_train_revised)
# x_test_revised=np.array(x_test_revised)
# y_test_revised=np.array(y_test_revised)

# # Defining the CNN model with regularization

# inputs = Input(shape=x_train.shape[1:])
# x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(5,5), activation='tanh',bias_regularizer=regularizers.L2(1e-4))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(5,5), activation='silu',activity_regularizer=regularizers.L2(1e-5))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid')(x)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(256, activation='softplus')(x)
# x = Dropout(rate=0.5)(x)
# outputs = Dense(10, activation='softmax')(x)
# net = Model(inputs=inputs, outputs=outputs)

# print(net.summary())
# plot_model(net, to_file='network_structure.png', show_shapes=True)
# net.compile(loss='categorical_crossentropy', optimizer='adam')

# # Fitting the model 

# history = net.fit(x_train_revised,y_train_revised,validation_data=(x_test_revised,y_test_revised),epochs=50,batch_size=250)

# # Plotting the error

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# net.save("Final_1.h5")
# net=load_model("Final_1.h5")

# outputs=net.predict(x_test)
# labels_predicted=np.argmax(outputs, axis=1)
# misclassified=sum(labels_predicted!=labels_test)
# print('Percentage misclassified = ',100*misclassified/labels_test.size)

# plt.figure(figsize=(8, 2))
# for i in range(0,8):
#     ax=plt.subplot(2,8,i+1)
#     plt.imshow(x_test[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r'))
#     plt.title(labels_test[i])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# for i in range(0,8):
#     output = net.predict(x_test[i,:].reshape(1, 28,28,1)) 
#     output=output[0,0:]
#     plt.subplot(2,8,8+i+1)
#     plt.bar(np.arange(10.),output)
#     plt.title(np.argmax(output))
#     plt.show()

# end = timeit.timeit()
# print("The time elapsed is ", end - start)

# # Hello sir

# # I would like to mention a few glitches I faced when I was implementing the file. I would like to mention them all to ensure that the application runs effectively in case you want to test it otherwise. 

# # --> Apart from the packages you had mentioned in the tutorial I had to install package which was pydot to render the graph. 
# # --> I had to set the environment variable 'KMP_DUPLICATE_LIB_OK']='True' to avoid error while running it 
# # --> I have preferred CNNs over MLPs because of their tolerance to position 
# # --> To improve generalization and tolerance to new values, drop out regularization has been used
# # --> Further, more types of regulaization which has not been mentioned in the tutorial such as kernel, bias and activity regularization have also been used.
# # --> To make sure the model is tolerant in size, phase and angle variations, Data augmentation using the Cut mix augmentation type has been implemented for preprocessing the data
# # Upon running for the test use cases, the percentage correctly classified was about 98.63 of the test data

# # Importing files 

# import numpy as np
# import keras
# import tensorflow
# import os 
# import timeit 
# import matplotlib.pyplot as plt 
# from tensorflow.keras.datasets import mnist
# from keras import layers
# from tensorflow.keras.models import load_model
# from tensorflow import clip_by_value
# from tensorflow import data as tf_data
# from tensorflow import image as tf_image
# from tensorflow import random as tf_random
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten,Dropout, Input
# from keras import regularizers
# from tensorflow.keras.utils import plot_model

# # Timing the entire exceution time 

# start = timeit.timeit()
# print("Started the timing")

# # Setting the envirnoment variables

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# keras.utils.set_random_seed(42)

# # Loading the data

# (x_train, labels_train), (x_test, labels_test) = mnist.load_data()
# y_train = keras.utils.to_categorical(labels_train, num_classes=10)
# y_test = keras.utils.to_categorical(labels_test, num_classes=10)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# # Data Augmentation

# AUTO = tf_data.AUTOTUNE
# BATCH_SIZE = 1
# IMG_SIZE = 28

# train_ds_one = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))
# train_ds_two = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))

# train_ds_simple = tf_data.Dataset.from_tensor_slices((x_train, y_train))
# test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test))

# train_ds_simple = (train_ds_simple.batch(BATCH_SIZE).prefetch(AUTO))
# train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

# def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
#     gamma_1_sample = tf_random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf_random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# def get_box(lambda_value):
#     cut_rat = tensorflow.math.sqrt(1.0 - lambda_value)

#     cut_w = IMG_SIZE * cut_rat 
#     cut_w = tensorflow.cast(cut_w, "int32")

#     cut_h = IMG_SIZE * cut_rat
#     cut_h = tensorflow.cast(cut_h, "int32")

#     cut_x = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # rx
#     cut_x = tensorflow.cast(cut_x, "int32")
#     cut_y = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # ry
#     cut_y = tensorflow.cast(cut_y, "int32")

#     boundaryx1 = clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
#     boundaryy1 = clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
#     bbx2 = clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
#     bby2 = clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

#     target_h = bby2 - boundaryy1
#     if target_h == 0:
#         target_h += 1

#     target_w = bbx2 - boundaryx1
#     if target_w == 0:
#         target_w += 1

#     return boundaryx1, boundaryy1, target_h, target_w


# def cutmix(train_ds_one, train_ds_two):
#     (image1, label1), (image2, label2) = train_ds_one, train_ds_two

#     alpha = [0.25]
#     beta = [0.25]

#     lambda_value = sample_beta_distribution(1, alpha, beta)

#     lambda_value = lambda_value[0][0]
#     boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

#     crop2 = tf_image.crop_to_bounding_box(image2, boundaryy1, boundaryx1, target_h, target_w)
#     image2 = tf_image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

#     crop1 = tf_image.crop_to_bounding_box(image1, boundaryy1, boundaryx1, target_h, target_w)


#     img1 = tf_image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
#     image1 = image1 - img1
#     image = image1 + image2

#     lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
#     lambda_value = tensorflow.cast(lambda_value, "float32")

#     label = lambda_value * label1 + (1 - lambda_value) * label2
#     return image, label

# train_ds_cmu = (train_ds.shuffle(1024).map(cutmix, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

# train_ds_cmu=list(train_ds_cmu)
# test_ds=list(test_ds)
# x_train_revised=[]
# y_train_revised=[]
# x_test_revised=[]
# y_test_revised=[]
# x_test_revised=[]
# y_test_revised=[]

# for i in range (60000):
#     x_train_revised.append(train_ds_cmu[i][0][0])
#     y_train_revised.append(train_ds_cmu[i][1][0])

# for i in range (10000):
#     x_test_revised.append(test_ds[i][0])
#     y_test_revised.append(test_ds[i][1])

# x_train_revised=np.array(x_train_revised)
# y_train_revised=np.array(y_train_revised)
# x_test_revised=np.array(x_test_revised)
# y_test_revised=np.array(y_test_revised)

# # Defining the CNN model with regularization

# inputs = Input(shape=x_train.shape[1:])
# x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(5,5), activation='tanh',bias_regularizer=regularizers.L2(1e-4))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(5,5), activation='silu',activity_regularizer=regularizers.L2(1e-5))(inputs)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid')(x)
# x = MaxPool2D(pool_size=(2, 2))(x)
# x = Flatten()(x)
# x = Dense(256, activation='softplus')(x)
# x = Dropout(rate=0.5)(x)
# outputs = Dense(10, activation='softmax')(x)
# net = Model(inputs=inputs, outputs=outputs)

# print(net.summary())
# plot_model(net, to_file='network_structure.png', show_shapes=True)
# net.compile(loss='categorical_crossentropy', optimizer='adam')

# # Fitting the model 

# history = net.fit(x_train_revised,y_train_revised,validation_data=(x_test_revised,y_test_revised),epochs=50,batch_size=250)

# # Plotting the error

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(history.history['loss'], label='training loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()

# net.save("Final_1.h5")
# net=load_model("Final_1.h5")

# outputs=net.predict(x_test)
# labels_predicted=np.argmax(outputs, axis=1)
# misclassified=sum(labels_predicted!=labels_test)
# print('Percentage misclassified = ',100*misclassified/labels_test.size)

# plt.figure(figsize=(8, 2))
# for i in range(0,8):
#     ax=plt.subplot(2,8,i+1)
#     plt.imshow(x_test[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r'))
#     plt.title(labels_test[i])
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# for i in range(0,8):
#     output = net.predict(x_test[i,:].reshape(1, 28,28,1)) 
#     output=output[0,0:]
#     plt.subplot(2,8,8+i+1)
#     plt.bar(np.arange(10.),output)
#     plt.title(np.argmax(output))
#     plt.show()

# end = timeit.timeit()
# print("The time elapsed is ", end - start)

# Hello sir

# I would like to mention a few glitches I faced when I was implementing the file. I would like to mention them all to ensure that the application runs effectively in case you want to test it otherwise. 

# --> Apart from the packages you had mentioned in the tutorial I had to install package which was pydot to render the graph. 
# --> I had to set the environment variable 'KMP_DUPLICATE_LIB_OK']='True' to avoid error while running it 
# --> I have preferred CNNs over MLPs because of their tolerance to position 
# --> To improve generalization and tolerance to new values, drop out regularization has been used
# --> Further, more types of regulaization which has not been mentioned in the tutorial such as kernel, bias and activity regularization have also been used.
# --> To make sure the model is tolerant in size, phase and angle variations, Data augmentation using the Cut mix augmentation type has been implemented for preprocessing the data
# Upon running for the test use cases, the percentage correctly classified was about 98.63 of the test data

# Importing files 

import numpy as np
import keras
import tensorflow
import os 
import timeit 
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist
from keras import layers
from tensorflow.keras.models import load_model
from tensorflow import clip_by_value
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import random as tf_random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten,Dropout, Input
from keras import regularizers
from tensorflow.keras.utils import plot_model

# Timing the entire exceution time 

start = timeit.timeit()
print("Started the timing")

# Setting the envirnoment variables

os.environ['KMP_DUPLICATE_LIB_OK']='True'
keras.utils.set_random_seed(42)

# Loading the data

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
y_train = keras.utils.to_categorical(labels_train, num_classes=10)
y_test = keras.utils.to_categorical(labels_test, num_classes=10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Data Augmentation

AUTO = tf_data.AUTOTUNE
BATCH_SIZE = 1
IMG_SIZE = 28

train_ds_one = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))
train_ds_two = (tf_data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1024))

train_ds_simple = tf_data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf_data.Dataset.from_tensor_slices((x_test, y_test))

train_ds_simple = (train_ds_simple.batch(BATCH_SIZE).prefetch(AUTO))
train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf_random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf_random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def get_box(lambda_value):
    cut_rat = tensorflow.math.sqrt(1.0 - lambda_value)

    cut_w = IMG_SIZE * cut_rat 
    cut_w = tensorflow.cast(cut_w, "int32")

    cut_h = IMG_SIZE * cut_rat
    cut_h = tensorflow.cast(cut_h, "int32")

    cut_x = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # rx
    cut_x = tensorflow.cast(cut_x, "int32")
    cut_y = tensorflow.random.uniform((1,), minval=0, maxval=IMG_SIZE)  # ry
    cut_y = tensorflow.cast(cut_y, "int32")

    boundaryx1 = clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
    boundaryy1 = clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
    bbx2 = clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
    bby2 = clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w


def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    lambda_value = sample_beta_distribution(1, alpha, beta)

    lambda_value = lambda_value[0][0]
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    crop2 = tf_image.crop_to_bounding_box(image2, boundaryy1, boundaryx1, target_h, target_w)
    image2 = tf_image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)

    crop1 = tf_image.crop_to_bounding_box(image1, boundaryy1, boundaryx1, target_h, target_w)


    img1 = tf_image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
    image1 = image1 - img1
    image = image1 + image2

    lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
    lambda_value = tensorflow.cast(lambda_value, "float32")

    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label

train_ds_cmu = (train_ds.shuffle(1024).map(cutmix, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO))

train_ds_cmu=list(train_ds_cmu)
test_ds=list(test_ds)
x_train_revised=[]
y_train_revised=[]
x_test_revised=[]
y_test_revised=[]
x_test_revised=[]
y_test_revised=[]

for i in range (60000):
    x_train_revised.append(train_ds_cmu[i][0][0])
    y_train_revised.append(train_ds_cmu[i][1][0])

for i in range (10000):
    x_test_revised.append(test_ds[i][0])
    y_test_revised.append(test_ds[i][1])

x_train_revised=np.array(x_train_revised)
y_train_revised=np.array(y_train_revised)
x_test_revised=np.array(x_test_revised)
y_test_revised=np.array(y_test_revised)

# Defining the CNN model with regularization

inputs = Input(shape=x_train.shape[1:])
x = Conv2D(filters=32, kernel_size=(3,3), activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=32, kernel_size=(5,5), activation='tanh',bias_regularizer=regularizers.L2(1e-4))(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=32, kernel_size=(5,5), activation='silu',activity_regularizer=regularizers.L2(1e-5))(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(256, activation='softplus')(x)
x = Dropout(rate=0.5)(x)
outputs = Dense(10, activation='relu')(x)
net = Model(inputs=inputs, outputs=outputs)

print(net.summary())
plot_model(net, to_file='network_structure.png', show_shapes=True)
net.compile(loss='categorical_crossentropy', optimizer='adam')

# Fitting the model 

history = net.fit(x_train_revised,y_train_revised,validation_data=(x_test_revised,y_test_revised),epochs=50,batch_size=250)

# Plotting the error

import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

net.save("Final_1.h5")
net=load_weights("Final_1.h5")

outputs=net.predict(x_test)
labels_predicted=np.argmax(outputs, axis=1)
misclassified=sum(labels_predicted!=labels_test)
print('Percentage misclassified = ',100*misclassified/labels_test.size)

plt.figure(figsize=(8, 2))
for i in range(0,8):
    ax=plt.subplot(2,8,i+1)
    plt.imshow(x_test[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r'))
    plt.title(labels_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for i in range(0,8):
    output = net.predict(x_test[i,:].reshape(1, 28,28,1)) 
    output=output[0,0:]
    plt.subplot(2,8,8+i+1)
    plt.bar(np.arange(10.),output)
    plt.title(np.argmax(output))
    plt.show()

end = timeit.timeit()
print("The time elapsed is ", end - start)

