This model predicts 43 different types of traffic signs on the German traffic 
sign recognition benchmark (gtsrb) dataset. 
gtsrb-small is a smaller version of the dataset, while the full gtsrb dataset is available at 
https://benchmark.ini.rub.de/gtsrb_dataset.html#Downloads

The purpose of this readme file is to explain the optimization of the "traffic.py" 
get_model function. Specifically, through the course of my experimentation, I had
two main goals in mind: 
1)the main parameter is maximum accuracy and minimum loss in my predictions
2)speed and minimizing time of computation is also important. 

In experimenting, I tested each of the variables one at a time, to make sure that
I could optimize each of the variables in a scientific manner. First, I tried changing 
the activation functions in each of the convolution, hidden, and final layers. When
I tried using 'sigmoid' or 'softmax' activation functions in my convolution and hidden sequence,
the accuracy was much worse than using a 'relu', I suspect that mathematically this
can be explained by the fact that a normal probability distribution represented by
'sigmoid' or 'softmax' may have a high variance in intermediate steps as opposed to relu,
which just remains 0 until it is activated. However, 'softmax' was definitely the best
activation function to use in the final labeling category and gave me the most accurate
results. 

Second, I added two layers of Convolution and MaxPooling sequences that shrink the image to
a 6 by 6 by 3 RGB matrix. Using one Convolution and Pooling sequence gave me similar accuracy
results, but still a little worse off than two. Generally, adding more Convolution and Pooling sequences
did not really affect computatation time. Also, adding one more convolution or pooling sequence
made my predictions slightly less accurate. This is probably due to the fact that these steps
would further shrink my final matrix, when it is really already small enough after two 
layers of these sequences ( 6 x 6 pixels).

After the convolution step, I tried experimenting with different numbers of hidden layers. I found
one hidden layer between the convolution and final output layer to be sufficient. Any more layers,
and the result became less accurate. This probably means that the best hypothetical function
to label the data is not very complex and can be modelled by probably nothing more than a quadratic or
second degree polynomial equation; this result can be expected for a 2-d image classification problem.

Finally, I tried to change the number of nodes in my hidden layer and the dropout rate. I increased 
the number of nodes by powers of two, and I realized that my maximum accuracy was achieved at around
256. Any more nodes increased the computation time by several seconds for each epoch, and I made only
miniscule improvements in accuracy and loss. Also, the dropout function was very helpful. I found
when I set the dropout rate to .5, I got a higher final accuracy compared to when I didn't use dropout. 
When the model was being trained, each of the training epochs showed lower accuracy and higher loss
when I used dropout compared to when I didn't, but these values underestimated the final accuracy and loss
when I used dropout and overestimated when I didn't. Dropping neurons out made the network as a whole more
resilient, and hence why I managed to get higher accuracy with dropout. 

Upon repeated attempts of running my model, I generally got an accuracy of around .985 - .99 , and a loss
of around .04-.05. 