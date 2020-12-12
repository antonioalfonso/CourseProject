# Text Classification Competition: Twitter Sarcasm Detection 
# Submission by Antonio Gomez Lopez

#This model was built using Neural Networks, as a consequence the training
#time can vary significantly depending on whether you train in a local
#environment (e.g. your computer), or GPUs (e.g. Google Colab)

If you want to run it using Colab, there is a read-only version of the notebook
available on the link below:

https://colab.research.google.com/drive/1kEjXNHX-tHkjG38BJtw9ol2QH67lGXaG?usp=sharing

Otherwise, use the WinnerClassificationCompetition.ipynb. I recommend using Jupyter,
but I assume any good Python editor would do the work.

1) What is required to run this model?

In both cases you need to have the files train.jsonl, and test.jsonl ready to be loaded.

-If you want to use the Google Colab instance, you should:
	1) mount your google drive (assumes you have a gmail or otherwise google-compatible account)
	2) upload the train.jsonl and test.jsonl at the root of your drive

As you run the code, you might get the following message:

"Warning: This notebook was not authored by Google.
This notebook was authored by aalfonzo@gmail.com. It may request access to your data stored 
with Google, or read data and credentials from other sessions. Please review the source code 
before executing this notebook. Please contact the creator of this notebook at aalfonzo@gmail.com 
with any additional questions."

-If you want to use the local run instance, you should:
	1) From the folder you are running the .ipynb file, you should have a /data folder, in which
	you will place both the train.jsonl and the test.jsonl files.

2) About the workbooks:

-Both workbooks (ipynb) have cell-by-cell runs with my comments on the code explaning either what the
code does or why did I choose specific parameter values.

-Remember that the local run instance will train much more slowly (it can take you many hours to get
to the trained model). For instance, in my laptop it took roughly 35 minutes per epoch, so 35x3=105 minutes,
a little bit less than two hours to have the model trained.

-Please note that:

1) The final design, fine-tuning and training of the network, including number of nodes, size of input,
batch-size, dropout rates, which optimizer and learning rate to use were all of my authorship.

2) Having said that, when it comes to the Neural Network solution I am presenting, I got inspired by two
other projects I found in Kaggle:

[A] Sarcasm detection using BERT (92% accuracy), by Raghav Khemka. 
https://www.kaggle.com/raghavkhemka/sarcasm-detection-using-bert-92-accuracy,

[B] Sarcasm detection with Bert ( val accuracy 98.5%), by Carmen SanDiego (clearly a pseudonym).
https://www.kaggle.com/carmensandiego/sarcasm-detection-with-bert-val-accuracy-98-5

[C] https://arxiv.org/pdf/1810.04805.pdf
To get information about best optimizers and learning rates to use for BERT

3) Also, important to note, in the course of the project I have worked with different 
pre-defined packages (and data transformations), that had made my views evolve (looking for higher
F1 scores). Below is a summary of what I worked with and the average F1 scores I got in each case.

Technology				F1 Score
- MetaPy (no features, Naive Bayes)	~57%
- Embeddings and GloVe			~66%
- Doc2Vec				~68%
- FastText				~70%
- Embeddings and BERT			~73.1%

Throughout the workbook you will find comments that seek to justify my choices many of the parameters
and fine-tuning completed. The most important things to know are:

1) The model worked best when running only for 3 epochs, given my selection of optimizer and learning rate.

2) I had to dropout roughly 40% of the nodes (it is considered normal to do dropouts between 20% and 50%. This
was the case because I was getting overfitting on the training data already in the third epoch (which indicates
that maybe my network was too big to begin with...).

3) I tried multiple times to provide a saved model so that you could run the network the same way I did, however
for unknown reasons, even though I was able to succesfully save my models, I was not able to load them back again.

4) As a consequence of 3), it is likely that a run of the model as it is might not render results that beat the 
baseline, (because is dependent on the training), so take that into consideration as you do the peer-review or
assessment. 