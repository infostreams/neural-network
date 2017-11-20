About
=====
The Tremani Neural Network allows you to build, train and employ neural networks in PHP. It is easy to use and set up, and does not rely on external software to be installed on your webserver. The software is open source under the BSD license, which means you can use and modify it freely.

About neural networks
---------------------
A neural network can be used to find complex relationships between data. Usually, you start with a large set of data that has some unknown relationship between input and output. A neural network can be used to find that unknown relationship. Once that relationship is found, the neural network can be used to compute the output for similar (but usually different) input. So, essentially, neural networks can learn complex relationships between input and output.

For example, neural networks can learn the XOR-function, it can be used to estimate the difficulty of a text or be trained in pattern recognition.

Previously advanced
-------------------
This software implements the 2007 state-of-the-art technology in neural networks. There has been immense progress in this field since that time though, but this software is still useful as an introduction to neural networks or if you need something fairly simple. The network that is created is a feed forward, multi-layer perceptron network with support for momentum learning and an advanced mechanism to prevent overfitting. Also, it allows you to easily adapt it to your needs by overriding key methods such as the activation function.

History
-------
This software was originally built in early 2007 for one of the projects we did at [Tremani](http://www.tremani.nl), a most excellent web design & web development agency based in charmingly beautiful Delft, the Netherlands. 

The goal of this project was to allow a user to determine the ‘[language reference level](http://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages)’ (i.e. the 'difficulty') of any given text. Basically, it will tell you if a given text is difficult to read or easy to understand, by looking at the complexity of the used language.

To determine this language reference level, we first did a statistical analysis of the text. Then, the resulting characteristics of the text are fed to a neural network. This neural network then applies the knowledge it has obtained in an earlier phase to determine the text’s difficulty.

We chose to build this system with a neural network because no comprehensive knowledge exists on the relationship between our input characteristics and the corresponding output. The relationship does exist, but is hard to find – and even harder to describe in software. However, a neural network can find and describe such a relationship quite easily.

Existing neural networks available for PHP at the time (most notably FANN) were difficult to set up, so we built one ourselves.

Documentation
-------------
There is some API level documentation available in the 'documentation.html' file, or just have a look at the provided example code.

Good luck! If you ever build something nice with this, please let me know.

