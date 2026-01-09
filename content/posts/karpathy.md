---
title: "Karpathy Summary"
date: 2025-12-23T10:00:00-05:00
draft: true
author: "Kartik"
tags: ["random","thoughts","karpathy"]
categories: ["misc"]
description: "Andrej Karpathy blog summary"
---

I have decided to speed-run all of the older [Karpathy blogs](https://karpathy.github.io/) and the latest one to extract and see their relevance now, along with interesting things we can ponder about



From [CV Blog]()

Turns out the image used in the CV blog still has GPT 5.2 in confusion. So even SOTA multimodal models are unable to extract ideas
from images like humans do


![Screenshot of Karpathy post](/images/karpathy.png)

Here is the output from GPT 5.2 Thinking mode  after prompting with 
"Why is this funny".

A couple specific visual gags are doing a lot of work:

Incongruity: “Serious people doing serious business”… but the center of attention is a bathroom/clinic scale in a hallway.

Body-language comedy: the guy in front has his leg stretched out and is leaning in with a grin, like he’s about to mess with the scale or playfully “help,” which feels out of place in such a formal setting.

The height contrast: the person on the scale looks especially tall on that platform, which exaggerates the whole moment.

Mirrors/reflections: you can see reflections on the left, which adds to the slightly absurd “caught in the act” vibe.


The model is unable to understand the mental play here -- that the person would be conscious about his weight and playing with the scale would change it -- effectively showing a larger value.
It also missed out that its the president doing this act of comedy.

The **emotional** and the thought processes that might occur with this are not caught by the LLM.

From [Feature Learning](https://karpathy.github.io/2014/07/03/feature-learning-escapades/)

He exclaims -- why do backbones trained from AlexNet really work for transfer learning. What is it exactly that enables this deep -transfer learning possible to other datasets. 

In short, why are representations learnt in one dataset transferrable to another one? What is this common manifold? Why does it exist in the first place. Clearly it must be a function of the data in the datasets -- if I pretrain AlexNet on images of dogs, would the same backbone be useful in predicting the type of flowers? -- Intuitively, No -- but this raises the question, what underlying latent does hold the property that the "backbone" is useful? Can I quantify the data manifold that is "correlated" to the pre-training dataset?

Another interesting discussion is why he believes things like unsupervised feature learning will not work -- quite simply because there is no bias in the learning process towards learning features of things we care about. Unsupervised learning seems to be doing a 
statistical loss minimization that is not at all directed about what the important aspects of the image are (eg. such as faces)
Doing things like auto-encoders makes the model follow statistical patterns in the data (for example, most of an image could be green grass, so the model learns great feature for grass and not other things).


He states humans learn in an unsupervised way, which is not completely right 

Humans learn via

- Prediction signals (self-supervised)

Your brain is constantly doing: “what will I see next?” “what will this look like from another angle?”
If your predictions are wrong, that’s an error signal.

This is basically self-supervised learning: predict missing/future sensory input.

- Temporal continuity and multi-view geometry

You see the same object across time while you move your head/eyes.
That gives a strong cue: “these two images are the same thing.”

This is a huge advantage over static image datasets.

- Action and embodiment (active learning)

You can poke, grab, rotate, walk closer. That’s not passive pixels—your actions create informative data.
You learn causality: “if I do A, B happens.”

-  Multi-modal alignment

Vision + sound + touch + proprioception + language later.
When a dog barks while you see it, those modalities reinforce each other.

-  Rewards and social feedback

Smiles, attention, approval/disapproval, success/failure at goals—these shape behavior and attention.
Not “labels,” but definitely supervision.

-  Built-in priors (structure from evolution)

Humans likely come with biases: face attention, object permanence tendencies, etc.
So the “learning algorithm” isn’t blank.

Adversarial robustness

It is interesting to read the section of how many sanity checks he makes when training models

"Digression: Technical fun parts. The fun part in actually doing this is that the standard AlexNetty ConvNet hyperparameters are of course completely inadequate. For example, normally you’d use weight decay of 0.0005 or so and learning rate of 0.01, and gaussian initialization drawn from a gaussian of 0.01 std. If you’ve trained linear classifiers before on this type of high-dimensional input (64x64x3 ~= 12K numbers), you’ll know that your learning rate will probably have to be much lower, the regularization much larger, and initialization of 0.01 std will probably be inadequate. Indeed, starting Caffe training with default hyperparameters gives a starting loss of about 80, which right away tells you that the initialization is completely out of whack (initial ImageNet loss should be ballpark 7.0, which is -log(1/1000)). I scaled it down to 0.0001 std for Gaussian init which gives sensible starting loss. But then the loss right away explodes which tells you that the learning rate is way too high - I had to scale it all the way down to about 1e-7. Lastly, a weight decay of 0.0005 will give almost negligible regularization loss with 12K inputs - I had to scale it up to 100 to start getting reasonably-looking weights that aren’t super-overfitted noise blobs. It’s fun being a Neural Networks practitioner."


RNNs post 

Sequence to sequence is one more type of model, but are there simply more tasks in the "input" to "output" sense that cannot be done via sequence to sequence models in the first place.
Does this give rich feature representations that we can use in the further steps that we can use for downstream tasks in the same way AlexNet features were used for downstream tasks in the first place?

Video classification per frame is an example of an immediate input-> output conversion per frame in the RNN.

Intuitive feeling of how much time each generation takes, and how the computation doubles with the size and dimension fo the hidden state



Deep RL

The four things that hold back 

Compute (the obvious one: Moore’s Law, GPUs, ASICs),
Data (in a nice form, not just out there somewhere on the internet - e.g. ImageNet),
Algorithms (research and ideas, e.g. backprop, CNN, LSTM), and
Infrastructure (software under you - Linux, TCP/IP, Git, ROS, PR2, AWS, AMT, TensorFlow, etc.).

