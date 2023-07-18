#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras


# In[14]:


(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


# In[15]:


x_train.shape, y_train.shape


# In[16]:


x_test.shape, y_test.shape


# In[17]:


x_train


# In[18]:


x_train[0]


# In[19]:


y_train[0]


# In[20]:


class_label = ["T-sirt/Top", "Trouser", "Pullover", "Dress","Coat", "Sandal", "Shirt", "Sneaker", "Bag",
'''
0 = > T-shirt/top
1 = > Trouser
2 = > Pullover
3 = > Dress  
4 = > Coat
5 = > Sandal
6 = > Shirt
7 = > Sneaker
8 = > Bag
9 = > Ankle boot'''
             ]


# In[21]:


plt.imshow(x_train[0], cmap = 'Greys')


# In[22]:


plt.imshow(x_train[1], cmap = 'Greys')


# In[23]:


y_test[1]


# In[24]:


plt.figure(figsize=(16,16))

j=1
for i in np.random.randint(0, 1000, 25):
    plt.subplot(5,5,j); j+=1
    plt.imshow(x_train[i], cmap="Greys")
    plt.axis('off')
    plt.title('{} / {}'.format(class_labels[y_train[i]], y_train[i]))


# In[25]:


x_train.ndim


# In[26]:


x_train.shape

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# In[27]:


x_train.ndim


# In[28]:


x_train.shape


# In[29]:


x_train = x_train/255.0
x_test = x_test/255


# In[30]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val, = train_test_split(x_train, y_train, test_size=0.2, random_state = 2)


# In[31]:


x_train.shape, y_train.shape


# In[32]:


x_val.shape, y_val.shape


# In[34]:


model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])


# In[35]:


model.summary()


# In[38]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[41]:


model.fit(x_train, y_train, epochs=2, batch_size=512, verbose=1, validation_data=(x_val,y_val))
#write epochs value 10 in colab


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




