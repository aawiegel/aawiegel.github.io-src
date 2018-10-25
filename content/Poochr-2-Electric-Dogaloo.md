Title: Poochr 2: Electric Dogaloo
Date: 2017-11-7
Category: Projects
Authors: Aaron Wiegel

<i>This is a follow up to a previous <a href="{filename}/Poochr.md">post</a>.</i>

## Overview

I was really dissatisfied with the performance of the topic modeling for Poochr's recommendations, so I decided to try a different approach for the text part. Based on examining the typical text input, I noticed people were usually focusing in on a few keywords like "apartment" or "children". So, I decided to try to use word embeddings such as [word2vec](https://code.google.com/archive/p/word2vec/) or [GloVe](https://nlp.stanford.edu/projects/glove/) instead. Word embeddings are typically trained by using the context in which a word appears to understand the meaning of the word. I used a pretrained GloVe model for this purpose because of the availability of more light weight models that I thought would be more suited to a responsive web app. Word embeddings are typically great for analogies but not so great at antonyms since words with opposite meanings tend to appear in the same context (like big dog or small dog.) Despite this shortcoming, the text part of the recommendations seemed perform better (...and was less sarcastic than topic modeling). However, the size of the word embedding model created some engineering issues, which I was able to overcome with some pruning down on the size of the vocabulary. I also made a few improvements to the interface, and I am continuing to monitor user input and feedback to see if further improvements can be made.

## Poochr's previous shortcomings

In the previous version of Poochr, I used topic modeling of a corpus describing dog breeds web scraped from dog websites and Wikipedia. I used topic modeling to generate vector representations of each breed topic and cosine similarity to user input to make recommendations. Unfortunately, although the documents were quite long, they contained a lot of "noise text" unrelated to traits people desire in dogs. (I don't think anyone cares that Pembroke Welsh Corgis were Queen Elizabeth II's favorite dog when selecting their next dog.) As a result, the recommender would often make "sarcastic" recommendations like [Burnese Mountain Dog](http://dogtime.com/dog-breeds/bernese-mountain-dog) when a user input "apartment".

<br/>

<img src="{filename}/images/burnese.jpg"/>

Pro-tip for Poochr: Not a dog that belongs in an apartment.

## GloVe and word embeddings

To circumvent this, I used a pretrained GloVe model instead. Here, GloVe was trained on Wikipedia and news articles so has a much better semantic understanding of the meaning of words. To train, GloVe creates a table of word-word co-occurances (probabilities) and uses this to generate a smaller column of numbers (vectors) that represents the meaning of the word. Due to the mathematics of how it produces these vectors (remember logarithms from algebra?), the differences between vectors represent the ratio of probabilities of that two words might occur together. Because of this, GloVe performs really well on finding word analogies and similarities between words but struggles with antonyms (much like word2vec). Since words with opposite meanings often appear in the same context (good dog vs. bad dog), these types models tend to think the words are similar when they actually have opposing meanings.

## Feature engineering: the return of dog2vec

To try to best create dog vectors and deal with the noisy text data, I more carefully created features to describe each dog breed with words. To do this, I used numerical 5-star ratings that one dog breed website gave each dog on various traits such as "tendancy to bark" or "kid friendly". For each star rating above 3, I added one word associated with that trait. Here, I avoided adding words when I did not think users would search for them. (I don't think anyone necessarily wants a dog that barks a lot.) For example, if a dog such as the Yorkshire Terrier had a 5-star rating for "energy level", I added the word "energetic" to the words describing the dog twice. I also did the same for star ratings below 3 when the trait might be relevant. For example, if a dog had a 2-star rating for "exercise needs", I added the word "lazy" to the words describing the dog once. I then went through each word and added the GloVe vector for each word up to construct a total dog word vector. In essence, I converted a number into words and then back into numbers again. (num2word2vec?)

<br/>

Here, the issue with antonyms and context based techniques like word2vec/GloVe were an important consideration. I tried to get around this issue by avoiding putting two directly opposing words as part of the dog word vectors such as "big" and "small". I used words that captured some of that same meaning like "apartment" to describe dogs who adapt well to apartment living (like *gasp* Chihuahuas) who are generally small. While this does not entirely circumvent the issue, I think it helped reduce the sarcastic recommendation problem I discussed above. (Prove me wrong by testing the [web app](https://poochr-182700.appspot.com/) out!)

## Engineering challenges in using word embeddings

One of the biggest problems with modifying Poochr to include word embeddings actually was more on the engineering side than on the data science side. Namely, compared to topic modeling, word embeddings are much more resource intensive even if you use a pre-trained model. Although I tried to use the smallest, 50-dimensional, 400,000 word vocabulary GloVe model, I kept getting timeout errors with my workers trying to load the models in Google App Engine. Based on looking at the error logs, the virtual machines were running out of RAM and thus trying to load everything with virtual memory, which is very slow in comparison.

<br/>

While I could have just increased the size of the virtual machines' RAM, I decided to try to come up with a low-weight version since the app already serves slowly as it is. Based on previous users' input, I checked how rare the words people used to describe their ideal dog. Almost all of the words were within the first 100,000 most common words in this GloVe model's vocabulary, so I trimmed the vocabulary of the GloVe model down to 100,000 words. This greatly reduced the size of the model, so I could then easily run Poochr again without much loss in the quality of the recommendations. If I wanted to further improve the speed of the recommendations, I would probably also try to retrain the image model with [Squeezenet](https://github.com/DeepScale/SqueezeNet), which is smaller than Xception and more suited to user-facing applications.

## Summary

Word embeddings can be a great way to get more semantic relationships with words, particularly analogies, but do run into trouble with antonyms. Compared to using topic modeling for recommendations, though, they are definitely superior in performance if you have a smaller number of documents where where a few keywords are important.  If you're considering using them in a user-facing application, be careful! Even the smaller pre-trained models are pretty large and may cause issues with memory. In any case, this was a fun project to work on, especially since I learned a lot about the user experience and engineering side of a data science project.

<br/>

The project code is available [here](https://github.com/aawiegel/Poochr).
