Title: Poochr: dog breed recommendations
Date: 2017-10-17
Category: Projects
Authors: Aaron Wiegel

# Dog breed recommendations

The [code](https://github.com/aawiegel/Poochr) is available online. This is part of a two part series where the second part can be found [here]({static}/Poochr-2-Electric-Dogaloo.md).

## Summary

Recently, my house mate decided to get a Siberian husky because it is a pretty dog. He didn't realize, though, that huskies are very social and incredibly active even compared to your typical dog. This temperament is not necessarily conducive to his lifestyle, so the situation has not been ideal for either him or the dog (or his house mates.) Based on this, I thought it would be a good idea to come up with a recommendation engine that suggests dogs based on both an image (appearance) and text description (temperament).

<br/>

<img src="{static}/images/bella.jpg" alt="a very active, vocal, and social dog" style="width: 100%;"/>

A rare moment where she is actually sitting still (getting a picture that wasn't a blur was the hardest part of this project.)

<br/>

To do this, I created vector representations (i.e., a column of numbers) of both a dog breed's appearance and temperament and combined them into one recommendation system. The representation of appearance came from a deep learning dog breed image classifier, and the representation of the temperament came from a dog breed topic model.  The resulting breed topic and image vectors were then compared to a user image and text to recommend the three most similar dogs. Purely appearance driven recommendations seemed to be more accurate but were not aggressive enough of labeling "not dog" images. Conversely, the text part was sometimes a bit off from data quality issues (some of the text I web scraped contradicted itself). Based on initial user feedback, I was able to improve the system's recommendations, To help improve the system in the future, I stored the user input in a database and solicited feedback on the quality of the recommendations and possible useful features.

## Overview

Most people get dogs because they want something cute, but the personality and temperament of the dog is an important factor for successful adoption into a human family. For my house mate's husky, she is quite active and craves a lot of social attention since huskies were bred to hunt in packs by Siberian natives. In contrast, my house mate does not exercise much and leaves the house frequently. The dog thus spends a lot of time by herself outside, which predictably leads to neurotic and sometimes destructive behavior. This behavior leads to a cycle where the dog cannot be trusted unsupervised indoors, so she spends even more time by herself outside. For example, I awoke one time in the middle of the night to find the husky by herself chewing on my house mate's glasses! Another time she tore up a bag of corn tortillas and several bags of grains, leaving a mess throughout the entire house. To help avoid this situation in the future, I thought to develop a dog breed recommender that relied on both appearance and temperament.

## dog2vec: converting dog data to vectors

As data sets for this recommendation system, I used the [Stanford Dogs image data set](http://vision.stanford.edu/aditya86/ImageNetDogs/) along with some web scraped images and breed descriptions from a few different websites. I developed the dog breed recommender using two models: an image classifier and a topic model. The image classifier was trained via transfer learning with the convolutional neural network Xception and weights from ImageNet. For each of the 114 dog breeds and the "not dog" images (from [CalTech](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)) included in the 50,000 image data set, I took the average of the last layer for the images in the test set to create the appearance representations. The topic model was developed from text scraped from several different dog websites and Wikipedia. I dubbed the process of converting both the image and text data into vector reprsentations "dog2vec". After creation of the dog vectors for each breed, a user-provided image and text description could be compared to each breed via cosine similarity.

### Training the image model

To start training the image model, I first focused on classifying "dog" and "not dog" images (not [not hotdog](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)). To do this, I started with Xception framework using the weights traiend on ImageNet, a 14 million image dataset with 20,000+ different classes. Because the layers earlier in the neural network have already been trained to recognize the essential features of images like edges, I froze all but the last two layers of the network and then retrained the model. In this way, the neural network is only re-learning how the final few sets of latent features need to be combined into a dog or anything "not dog". In addition, when feeding images into the model, I used [Keras's preprocessing features](https://keras.io/preprocessing/image/) to randomly rotate, shear, and zoom the images to help deal with imperfect images that may be provided by users. With only 5 training epochs, the model was able to get 99.7% accuracy on distinguishing dog images from other images in the test set.

<br/>

Afterwards, I then divided the dog images into the 114 breeds and retrained with the weights of the last two layers from the dog-notdog model. Because the classes were so imbalanced (generally ~100 image per breed vs. 20,000 images in not dog in the training data), I applied class weights based on the frequency of images. Therefore, correctly identifying a "not dog" image was not weighted as heavily in the loss function (categorical cross entropy) as identifying an image of a dog breed. Without accounting for the class weights, the model would only correctly identify "not dog" images and essentially would randomly guess which dog breed the image represented. Consequently, the accuracy would get stuck at the proportion of "not dog" images in the test data set (~60%).

<br/>

After using class weights, the accuracy rose to about 92% on the test set, although the model was much better at identifying some breeds than others. For example, the model identified the Samoyed breed correctly 98.7% of the time!

<br/>

<img src="{static}/images/samoyed.jpg" style="width: 100%;"/>

Apparently Poochr likes really big, poofy white dogs.

<br/>

Other dog breeds, particularly ones that looked very similar to each other, could not be so easily identified. For example, Australian terriers (21% accuracy) and Yorkshire terriers (31% accuracy) look very similar:

<img src="{static}/images/australian_terrier.jpg" alt="Australian Terrier" style="width:50%;"/><img src="{static}/images/yorkshire_terrier.jpg" alt="Yorkshire Terrier" style="width=50%;"/>

Can you tell which is which? Yeah, the model can't really either.

<br/>

In general, things like size differences are not apparent from images, so for breeds where the main difference is size, the image classification had a difficult time identifying the breed. A good example of this is the Schnauzer, where there are three different breeds based on the overall size of the dog: miniature, standard, and giant. The model tended to guess between all three, although it was slightly biased in favor of miniature schnauzers (easier to clean up after...?).

<br/>

Thankfully, since I was trying to develop a recommendation system, being able to identify the three most similar dog breeds was more important than perfectly classifying the dogs. In order to do the comparison, I first created a function using TensorFlow to calculate the final layer of the neural network before the softmax for classification. I averaged the final layer for each of the dog breed images in the test set. Then, I ran the user image through this same function to generate a dog vector. By finding the cosine similarity between the average breed vector and the user image vector, I was able to recommend the three closest dogs based on appearance. The final vector was also tested to see if the "not dog" component was the largest in the vector and gives a warning to the user if it suspects the image is not a dog.

### Topic modeling of dog breeds

To generate the temperament based recommendations, I web scraped text about dog breeds and did topic modeling on the resulting word frequency matrix for each dog breed. Since my purpose was to generate similarity scores based on user input, I thought LSA ([Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)) would be a good choice. Even though LSA does not necessarily truly produce real "topics" in any human sense of the word, it is a great way to find similarities between text. Other methods such as [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) and [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) will tend to produce topics that seem more natural to humans but do not necessarily provide better performance when comparing similarity. I still trained the other two topic models, but I used LSA as an initial baseline. I performed cosine similarity on the user provided text, and the similarity between the image and text were equally weighted.

## Initial user feedback

I deployed Poochr as a [web app](https://poochr-182700.appspot.com/) using Google Cloud and Flask. Using Flask, I was able to create an effective if fairly simple REST API to have allow users to upload images and text. I stored the user images, text, provided recommendations, and other data to evaluate the performance of the model. In particular, since a recommendation engine is an unsupervised problem, constantly tweaking and evaluating the model is incredibly important. I also solicited feedback from my initial test users (mostly friends and family.) Even in the initial deployment, I was able to discover some things I was able to fix about Poochr's recommendations:

<ul>
  <li>Initially, I thought to use the maximum of each component in the dog image vectors. This turned out to over-recommend Samoyed since the image classification algorithm often mistakenly identified dogs as Samoyed. After switching to the average of the final dog image layer, the recommendations were able to get much closer to the actual dog the user provided.</li>

  <br/>

  <li>The not dog part of the model was not nearly as aggressive as I would have liked. Namely, my friend went to the zoo recently and had a lot of photos on his phone of different animals. It had trouble properly telling the user that those animals were not dogs. I hope to remedy this in the future by using the images I stored to re-train the image classification algorithm.</li>

  <br/>

  <li>The quality of the text sources I used was somewhat dubious. In playing around with the model, one of my friends noticed that the text in one of the recommended dogs contradicted itself. (It said it was a quiet dog in one part and yappy in another part.) Here, I think I need to find a higher quality corpus to based my recommendations on. The American Kennel Club has a 400 page <a href="https://www.amazon.com/Encyclopedia-Breeds-Caroline-Coile-Ph-D-ebook/dp/B014PNWKP0/">book</a> describing dog breeds that would likely be a higher quality source.</li>
</ul>

Other features that came up that users were interested in were allowing URLs for images instead of just uploading, providing stock dog images in case you did not have one, and being able to adjust the recommendation based on how much to weight the appearance vs. temperament. I'd also personally like to add a way for users to input feedback more directly. With a better quality corpus, I'd also like to evaluate the performance between topic models, especially since LDA tends to perform better with smaller but longer documents. With better recommendations, hopefully more people can avoid the situation my house mate is in!

This is part of a two part series where the second part can be found [here]({static}/Poochr-2-Electric-Dogaloo.md).
