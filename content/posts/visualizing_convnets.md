---
title: "Visualizing parts of Convolutional Neural Networks using Keras and Cats"
date: 2017-01-22T13:05:42-07:00
draft: false
---

It is well known that convolutional neural networks (CNNs or ConvNets) have been the source of many major breakthroughs in the field of Deep learning in the last few years, but they are rather unintuitive to reason about for most people. I’ve always wanted to break down the parts of a ConvNet and see what an image looks like after each stage, and in this post I do just that!
CNNs at a high level
First off, what are ConvNets good at? ConvNets are used primarily to look for patterns in an image. You did that by convoluting over an image and looking for patterns. In the first few layers of CNNs the network can identify lines and corners, but we can then pass these patterns down through our neural net and start recognizing more complex features as we get deeper. This property makes CNNs really good at identifying objects in images.

## What is a CNN?
A CNN is a neural network that typically contains several types of layers, one of which is a **convolutional layer**, as well as **pooling**, and **activation** layers.

## Convolutional Layer
To understand what a CNN is, you need to understand how convolutions work. Imagine you have an image represented as a 5x5 matrix of values, and you take a 3x3 matrix and slide that 3x3 window around the image. At each position the 3x3 visits, you matrix multiply the values of your 3x3 window by the values in the image that are currently being covered by the window. This results in a single number the represents all the values in that window of the image. Here’s a pretty gif for clarity:

![Conv gif](/blog/convcats/conv1.gif)

As you can see, each item in the feature matrix corresponds to a section of the image. Note that the value of the kernel matrix is the red number in the corner of the gif.

The “window” that moves over the image is called a **kernel**. Kernels are typically square and 3x3 is a fairly common kernel size for small-ish images. The distance the window moves each time is called the stride. Additionally of note, images are sometimes padded with zeros around the perimeter when performing convolutions, which dampens the value of the convolutions around the edges of the image (the idea being typically the center of photos matter more).

The goal of a convolutional layer is **filtering**. As we move over an image we effective check for patterns in that section of the image. This works because of filters, stacks of weights represented as a vector, which are multiplied by the values outputed by the convolution.When training an image, these weights change, and so when it is time to evaluate an image, these weights return high values if it thinks it is seeing a pattern it has seen before. The combinations of high weights from various filters let the network predict the content of an image. This is why in CNN architecture diagrams, the convolution step is represented by a box, not by a rectangle; the third dimension represents the filters.

![Alexnet](/blog/convcats/alexnet.jpeg)

### Things to note:
- The output of the convolution is smaller (in width and height) than the original image
- A linear function is applied between the kernel and the image window that is under the kernel
- Weights in the filters are learned by seeing lots of images

## Pooling Layers

Pooling works very much like convoluting, where we take a **kernel** and move the kernel over the image, the only difference is the function that is applied to the kernel and the image window isn’t linear.

**Max pooling** and **Average pooling** are the most common pooling functions. Max pooling takes the largest value from the window of the image currently covered by the kernel, while average pooling takes the average of all values in the window.

![pooling gif](/blog/convcats/pooling.gif)

## Activation Layers
Activation layers work exactly as in other neural networks, a value is passed through a function that squashes the value into a range. Here’s a bunch of common ones:

![activation](/blog/convcats/activation.png)

The most used activation function in CNNs is the relu (Rectified Linear Unit). There are a bunch of reason that people like relus, but a big one is because they are really cheap to perform, if the number is negative: zero, else: the number. Being cheap makes it faster to train networks.

### Recap
- Three main types of layers in CNNs: **Convolutional, Pooling, Activation**
- **Convolutional layers** multiply kernel value by the image window and optimize the kernel weights over time using gradient descent
- **Pooling layers** describe a window of an image using a single value which is the max or the average of that window
- **Activation layers** squash the values into a range, typically [0,1] or [-1,1]

## What does a CNN look like?
Before we get into what a CNN looks like, a little bit of background. The first successful applications of ConvNets was by Yann LeCun in the 90’s, he created something called LeNet, that could be used to read hand written numbers. Since then, computing advancements and powerful GPUs have allowed researchers to be more ambitious.

In 2010 the Stanford Vision Lab released ImageNet. Image net is data set of 14 million images with labels detailing the contents of the images. It has become one of the research world’s standards for comparing CNN models, with current best models will successfully detect the objects in 94+% of the images. Every so often someone comes in and beats the all time high score on imagenet and its a pretty big deal. In 2014 it was GoogLeNet and VGGNet, before that it was ZF Net.

The first viable example of a CNN applied to imagenet was AlexNet in 2012, before that researches attempted to use traditional computer vision techiques, but AlexNet outperformed everything else up to that point by ~15%.

Anyway, lets look at LeNet:

![lenet](/blog/convcats/lenet.png)

This diagram doesn’t show the activation functions, but the architecture is:
Input image →ConvLayer →Relu → MaxPooling →ConvLayer →Relu→ MaxPooling →Hidden Layer →Softmax (activation)→output layer

## On to the cats!

Here is an image of a cat:

![cat](/blog/convcats/cat1.png)

Our picture of the cat has a height 320px, a width of 400px, and 3 channels of color (RGB).

## Convolutional Layer

So what does he look like after one layer of convolution?

![cat](/blog/convcats/cat2.png)

Here is the cat with a kernel size of 3x3 and 3 filters (if we have more than 3 filter layers we cant plot a 2d image of the cat. Higher dimensional cats are notoriously tricky to deal with.).

As you can see the cat is really noisy because all of our weights are randomly initialized and we haven’t trained the network. Oh and they’re all on top of each other so even if there was detail on each layer we wouldn’t be able to see it. But we can make out areas of the cat that were the same color like the eyes and the background. What happens if we increase the kernel size to 10x10?

![cat](/blog/convcats/cat2.png)

As we can see, we lost some of the detail because the kernel was too big. Also note the shape of the image is slightly smaller because of the larger kernel, and because math governs stuff.

What happens if we squish it down a bit so we can see the color channels better?

![cat](/blog/convcats/cat4.png)

Much better! Now we can see some of the things our filter is seeing. It looks like red is really liking the black bits of the nose an eyes, and blue is digging the light grey that outlines the cat. We can start to see how the layer captures some of the more important details in the photo.

![cat](/blog/convcats/cat5.png)

![cat](/blog/convcats/cat6.png)

![cat](/blog/convcats/cat7.png)

If we increase the kernel size its far more obvious now that we get less detail, but the image is also smaller than the other two.

### Add an Activation layer

![cat](/blog/convcats/cat8.png)

We get rid of of a lot of the not blue-ness by adding a relu.

### Adding a Pooling Layer

We add a pooling layer (getting rid of the activation just max it a bit easier to show)

![cat](/blog/convcats/cat9.png)

As expected, the cat is blockier, but we can go even blockyier!

![cat](/blog/convcats/cat10.png)

Notice how the image is now about a third the size of the original.

### Activation and Max Pooling

![cat](/blog/convcats/cat11.png)

### LeNet Cats

What do the cats look like if we put them through the convolutional and pools sections of LeNet?

![cat](/blog/convcats/cat12.png)

1 filter for each conv layer

![cat](/blog/convcats/cat13.png)

3 filters in the first conv layer, 1 in second conv layer

![cat](/blog/convcats/cat14.png)

3 filter layers in each convolution

## Conclusion

ConvNets are powerful due to their ability to extract the core features of an image and use these features to identify images that contain features like them. Even with our two layer CNN we can start to see the network is paying a lot of attention to regions like the whiskers, nose, and eyes of the cat. These are the types of features that would allow the CNN to differentiate a cat from a bird for example.

CNNs are remarkably powerful, and while these visualizations aren’t perfect, I hope they can help people like myself who are still learning to reason about ConvNets a little better.

All code is on Github: [https://github.com/erikreppel/visualizing_cnns](https://github.com/erikreppel/visualizing_cnns)

Follow me on Twitter, I’m [@programmer](https://twitter.com/programmer) (yes, seriously).

### Further Resources

[Andrej Karpathy’s cs231n](https://github.com/erikreppel/visualizing_cnns)

[A guide to convolution arithmetic for deep learning by Vincent Dumoulin and Francesco Visin](https://arxiv.org/pdf/1603.07285v1.pdf)
