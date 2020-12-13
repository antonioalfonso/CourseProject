# Text Classification Competition: Twitter Sarcasm Detection 
# Submission by Antonio Gomez Lopez

This model was built using Neural Networks, as a consequence the training time 
can vary significantly depending on whether you train in a local environment 
(e.g. your computer), or GPUs (e.g. Google Colab)

IMPORTANT: You can find the tutorial here (I show it running Colab and beat the baseline live :-))

https://www.youtube.com/watch?v=p1T-ekliduA

If you want to run it using Colab, there is a read-only version of the notebook
available on the link below:

https://colab.research.google.com/drive/1kEjXNHX-tHkjG38BJtw9ol2QH67lGXaG?usp=sharing

Otherwise, use the ClassificationCompetition_BERT_AntonioGomez.ipynb and run it locally. I recommend 
using Jupyter, but I assume any good Python editor would do the work. Check 2) to make sure you have
both train.jsonl and test.jsonl files in the right directory.

1) What's included in this submission?

1.1) This file includes details on what needs to be setup.
1.2) The reference to Google Colab (above) as well as the .ipynb file contain the commented
source code.
1.3) answer.txt contains the test set predictions (as required in https://docs.google.com/document/d/13ANy7FHYovh_2JL3gVrVvzXScDh5ol5l5XS2Nlp4DN4/edit)


2) What is required to run this model?

In both cases you need to have the files train.jsonl, and test.jsonl ready to be loaded.

2.1) If you want to use the Google Colab instance, you should:
	2.1.1) mount your google drive (assumes you have a gmail or otherwise google-compatible account)
	2.1.2) upload the train.jsonl and test.jsonl at the root of your drive

As you run the code, you might get the following message:

"Warning: This notebook was not authored by Google.
This notebook was authored by aalfonzo@gmail.com. It may request access to your data stored 
with Google, or read data and credentials from other sessions. Please review the source code 
before executing this notebook. Please contact the creator of this notebook at aalfonzo@gmail.com 
with any additional questions."

2.2)If you want to use the local run instance, you should:
	2.2.1) From the folder you are running the .ipynb file, you should have a /data folder, in which
	you will place both the train.jsonl and the test.jsonl files.

3) About the workbooks:

3.1) Both workbooks (ipynb) have cell-by-cell runs with my comments on the code explaning either what the
code does or why did I choose specific parameter values.

-Remember that the local run instance will train much more slowly (it can take you many hours to get
to the trained model). For instance, in my laptop it took roughly 35 minutes per epoch, so 35x3=105 minutes,
a little bit less than two hours to have the model trained.

4) Please note that:

4.1) The final design, fine-tuning and training of the network, including number of nodes, size of input,
batch-size, dropout rates, which optimizer and learning rate to use were all of my authorship after several
tests being performed.

4.2) Having said that, when it comes to the Neural Network solution I am presenting, I got inspired by two
other projects I found in Kaggle, and that I am referencing here:

[A] Sarcasm detection using BERT (92% accuracy), by Raghav Khemka. 
https://www.kaggle.com/raghavkhemka/sarcasm-detection-using-bert-92-accuracy,

[B] Sarcasm detection with Bert (val accuracy 98.5%), by Carmen SanDiego (clearly a pseudonym).
https://www.kaggle.com/carmensandiego/sarcasm-detection-with-bert-val-accuracy-98-5

[C] https://arxiv.org/pdf/1810.04805.pdf
To get information about best optimizers and learning rates to use for BERT

4.3) How did I arrive to the succesful model?

Background: Important to note, in the course of the project I have worked with different 
pre-defined packages (and data transformations), that I am not including here for the sake of
focus on what's really the working solution. Having said that, however, the exploring of different
alternatives was what had made my views evolve (looking for higher F1 scores). 

Below is a summary of what I worked with and the average F1 scores I got in each case.

Technology				F1 Score
- MetaPy (no features, Naive Bayes)	~57%
- Embeddings and GloVe			~66%
- Embeddings and Doc2Vec		~68%
- Embeddings and FastText (by Facebook)	~70%
- Embeddings and BERT			~73.1%

Note that the setup of each model was dependent on the pre-defined package I was using.

Explanation of the model:

It is based on a Neural Network with 5 layers (including the input and output layers, and
including one layer with embeddings provided by BERT).

I included a chance to Dropout on the basis of potential overfitting (having a too big network 
for the task at hand). In addition, Optimizer, learning rate, batch size, and loss functions were 
ALL parameters up for fine-tuning.

There is some pre-processing of the data: Mostly removed punctuation, special symbols and
stopwords. Having said that, I obtained best results when I did not removed @USER, <URL> 
references or emoticons.

Some encoding / padding was included to keep consistent the size of the input for tweets either
in training or testing.

How did the training happen?

For all the methods used (except Metapy - Naive Bayes), all of my training and initial fine-tuning happened 
first with the optimizer, learning rate, and "size" of embeddings (this last one was not the case for BERT).
I also experimented with pre-processing of data, but as I mentioned earlier, I seem to obtained better results
when I did not remove as many references that I originally thought would be mostly noise (e.g. @USER, <URL>).

In the specific case of BERT I performed the following fine-tuning (not necessarily in this order):

- Optimizer: Started with Adam, then moved to AdamWeightDecay, and finally got best results with Adamax.
As per the documentation in Tensorflow, Adamax is recommended when the model includes an embeddings layer.

- Learning rate and number of epochs: I used 2e-5 as direct recommendation of the BERT paper: https://arxiv.org/pdf/1810.04805.pdf
number of epochs is also recommended in this paper. I experimented however with large numbers (large as 10), 
but it was very clear that I was overfitting the model with the training data. I found that a number of epochs
between 2-4 would work best, but needed additional fine-tuning.

- Dropout rate: From the referenced model, they use a dropout rate of 20%. I was still overfitting in many
instances after the third epoch, so ended up increasing the droupout to 40%, and with that I could mitigate
the problem of overfitting the model with the training model.

- Pre-processing data: I coded different functions to remove different parts of the tweets that I thought
would constitute noise, but ended commenting most of them and leaving the functions that would remove just 
the stopwords and special punctuation symbols.

- Batch-size: I experimented with different batch sizes making sure that I was maximizing the maximum input
from BERT (512 bytes). I ended up with an input of 64 and batch size of 8, but I equally beat the baseline
with input size of 128 and batch size of 4.

5) Other important details about this implementation:

Throughout the workbook you will find comments that seek to justify my choices many of the parameters
and fine-tuning completed. The most important things to know are:

5.1) The model worked best when running only for 3 epochs, given my selection of optimizer and learning rate.

5.2) I had to dropout roughly 40% of the nodes (according to literature I consulted it is considered normal 
to do dropouts between 20% and 50%. This was the case because I was getting overfitting on the training data 
already in the third epoch (which indicates that maybe my network was too big to begin with...).

5.3) I tried multiple times to provide a saved model so that you could run the network the same way I did, however
for unknown reasons, even though I was able to succesfully save my models, I was not able to load them back again.

5.4) As a consequence of 5.3), it is likely that a run of the model as it is might not render results that beat the 
baseline, (because is dependent on the training), so take that into consideration as you do the peer-review or
assessment.

4.5) Having said that, I included my best run as part of the project submission (also as it was required as per
homework guidelines (see https://docs.google.com/document/d/13ANy7FHYovh_2JL3gVrVvzXScDh5ol5l5XS2Nlp4DN4/edit)