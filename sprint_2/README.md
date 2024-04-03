
# Sequence Models

## Table of contents
* [Sequence Models](#sequence-models)
   * [Table of contents](#table-of-contents)
   * [Recurrent Neural Networks](#recurrent-neural-networks)
      * [Why sequence models](#why-sequence-models)
      * [Notation](#notation)
      * [Recurrent Neural Network Model](#recurrent-neural-network-model)
      * [Backpropagation through time](#backpropagation-through-time)
      * [Different types of RNNs](#different-types-of-rnns)
      * [Language model and sequence generation](#language-model-and-sequence-generation)
      * [Sampling novel sequences](#sampling-novel-sequences)
      * [Vanishing gradients with RNNs](#vanishing-gradients-with-rnns)
      * [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
      * [Long Short Term Memory (LSTM)](#long-short-term-memory-lstm)
      * [Bidirectional RNN](#bidirectional-rnn)
      * [Deep RNNs](#deep-rnns)
      * [Back propagation with RNNs](#back-propagation-with-rnns)
   * [Sequence models &amp; Attention mechanism](#sequence-models--attention-mechanism)
      * [Various sequence to sequence architectures](#various-sequence-to-sequence-architectures)
         * [BLEU Score](#bleu-score)
         * [Attention Model Intuition](#attention-model-intuition)
         * [Attention Model](#attention-model)
   * [Extras](#extras)
      * [Machine translation attention model (From notebooks)](#machine-translation-attention-model-from-notebooks)

## Recurrent Neural Networks

> Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Why sequence models
- Sequence Models like RNN and LSTMs have greatly transformed learning on sequences in the past few years.
- Examples of sequence data in applications:
  - Speech recognition (**sequence to sequence**):
    - X: wave sequence
    - Y: text sequence
  - Music generation (**one to sequence**):
    - X: nothing or an integer
    - Y: wave sequence
  - Sentiment classification (**sequence to one**):
    - X: text sequence
    - Y: integer rating from one to five
  - DNA sequence analysis (**sequence to sequence**):
    - X: DNA sequence
    - Y: DNA Labels
  - Machine translation (**sequence to sequence**):
    - X: text sequence (in one language)
    - Y: text sequence (in other language)
  - Video activity recognition (**sequence to one**):
    - X: video frames
    - Y: label (activity)
  - Name entity recognition (**sequence to sequence**):
    - X: text sequence
    - Y: label sequence
    - Can be used by seach engines to index different type of words inside a text.
- All of these problems with different input and output (sequence or not) can be addressed as supervised learning with label data X, Y as the training set.

### Notation
- In this section we will discuss the notations that we will use through the course.
- **Motivating example**:
  - Named entity recognition example:
    - X: "Harry Potter and Hermoine Granger invented a new spell."
    - Y:   1   1   0   1   1   0   0   0   0
    - Both elements has a shape of 9. 1 means its a name, while 0 means its not a name.
- We will index the first element of x by x<sup><1></sup>, the second x<sup><2></sup> and so on.
  - x<sup><1></sup> = Harry
  - x<sup><2></sup> = Potter
- Similarly, we will index the first element of y by y<sup><1></sup>, the second y<sup><2></sup> and so on.
  - y<sup><1></sup> = 1
  - y<sup><2></sup> = 1

- T<sub>x</sub> is the size of the input sequence and T<sub>y</sub> is the size of the output sequence.
  
  - T<sub>x</sub> = T<sub>y</sub> = 9 in the last example although they can be different in other problems.
- x<sup>(i)\<t></sup> is the element t of the sequence of input vector i. Similarly y<sup>(i)\<t></sup> means the t-th element in the output sequence of the i training example.
- T<sub>x</sub><sup>(i)</sup> the input sequence length for training example i. It can be different across the examples. Similarly for T<sub>y</sub><sup>(i)</sup> will be the length of the output sequence in the i-th training example.

- **Representing words**:
    - We will now work in this course with **NLP** which stands for natural language processing. One of the challenges of NLP is how can we represent a word?

    1. We need a **vocabulary** list that contains all the words in our target sets.
        - Example:
            - [a ... And   ... Harry ... Potter ... Zulu]
            - Each word will have a unique index that it can be represented with.
            - The sorting here is in alphabetical order.
        - Vocabulary sizes in modern applications are from 30,000 to 50,000. 100,000 is not uncommon. Some of the bigger companies use even a million.
        - To build vocabulary list, you can read all the texts you have and get m words with the most occurrence, or search online for m most occurrent words.
    2. Create a **one-hot encoding** sequence for each word in your dataset given the vocabulary you have created.
        - While converting, what if we meet a word thats not in your dictionary?
        - We can add a token in the vocabulary with name `<UNK>` which stands for unknown text and use its index for your one-hot vector.
    - Full example:   
        ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//01.png)

- The goal is given this representation for x to learn a mapping using a sequence model to then target output y as a supervised learning problem.

### Recurrent Neural Network Model
- Why not to use a standard network for sequence tasks? There are two problems:
  - Inputs, outputs can be different lengths in different examples.
    - This can be solved for normal NNs by paddings with the maximum lengths but it's not a good solution.
  - Doesn't share features learned across different positions of text/sequence.
    - Using a feature sharing like in CNNs can significantly reduce the number of parameters in your model. That's what we will do in RNNs.
- Recurrent neural network doesn't have either of the two mentioned problems.
- Lets build a RNN that solves **name entity recognition** task:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//02.png)
  - In this problem T<sub>x</sub> = T<sub>y</sub>. In other problems where they aren't equal, the RNN architecture may be different.
  - a<sup><0></sup> is usually initialized with zeros, but some others may initialize it randomly in some cases.
  - There are three weight matrices here: W<sub>ax</sub>, W<sub>aa</sub>, and W<sub>ya</sub> with shapes:
    - W<sub>ax</sub>: (NoOfHiddenNeurons, n<sub>x</sub>)
    - W<sub>aa</sub>: (NoOfHiddenNeurons, NoOfHiddenNeurons)
    - W<sub>ya</sub>: (n<sub>y</sub>, NoOfHiddenNeurons)
- The weight matrix W<sub>aa</sub> is the memory the RNN is trying to maintain from the previous layers.
- A lot of papers and books write the same architecture this way:  
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//03.png)
  - It's harder to interpret. It's easier to roll this drawings to the unrolled version.
- In the discussed RNN architecture,  the current output y&#770;<sup>\<t></sup> depends on the previous inputs and activations.
- Let's have this example 'He Said, "Teddy Roosevelt was a great president"'. In this example Teddy is a person name but we know that from the word **president** that came after Teddy not from **He** and **said** that were before it.
- So limitation of the discussed architecture is that it can not learn from elements later in the sequence. To address this problem we will later discuss **Bidirectional RNN**  (BRNN).
- Now let's discuss the forward propagation equations on the discussed architecture:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//04.png)
  - The activation function of a is usually tanh or ReLU and for y depends on your task choosing some activation functions like sigmoid and softmax. In name entity recognition task we will use sigmoid because we only have two classes.
- In order to help us develop complex RNN architectures, the last equations needs to be simplified a bit.
- **Simplified RNN notation**:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//05.png)
  - w<sub>a</sub> is w<sub>aa</sub> and w<sub>ax</sub> stacked horizontaly.
  - [a<sup>\<t-1></sup>, x<sup>\<t></sup>] is a<sup>\<t-1></sup> and x<sup>\<t></sup> stacked verticaly.
  - w<sub>a</sub> shape: (NoOfHiddenNeurons, NoOfHiddenNeurons + n<sub>x</sub>)
  - [a<sup>\<t-1></sup>, x<sup>\<t></sup>] shape: (NoOfHiddenNeurons + n<sub>x</sub>, 1)

### Backpropagation through time
- Let's see how backpropagation works with the RNN architecture.
- Usually deep learning frameworks do backpropagation automatically for you. But it's useful to know how it works in RNNs.
- Here is the graph:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//06.png)
  - Where w<sub>a</sub>, b<sub>a</sub>, w<sub>y</sub>, and b<sub>y</sub> are shared across each element in a sequence.
- We will use the cross-entropy loss function:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//07.png)
  - Where the first equation is the loss for one example and the loss for the whole sequence is given by the summation over all the calculated single example losses.
- Graph with losses:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//08.png)
- The backpropagation here is called **backpropagation through time** because we pass activation `a` from one sequence element to another like backwards in time.

### Different types of RNNs
- So far we have seen only one RNN architecture in which T<sub>x</sub> equals T<sub>Y</sub>. In some other problems, they may not equal so we need different architectures.
- The ideas in this section was inspired by Andrej Karpathy [blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). Mainly this image has all types:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//09.jpg)
- The architecture we have descried before is called **Many to Many**.
- In sentiment analysis problem, X is a text while Y is an integer that rangers from 1 to 5. The RNN architecture for that is **Many to One** as in Andrej Karpathy image.   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//10.png)
- A **One to Many** architecture application would be music generation.  
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//11.png)
  - Note that starting the second layer we are feeding the generated output back to the network.
- There are another interesting architecture in **Many To Many**. Applications like machine translation inputs and outputs sequences have different lengths in most of the cases. So an alternative _Many To Many_ architecture that fits the translation would be as follows:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//12.png)
  - There are an encoder and a decoder parts in this architecture. The encoder encodes the input sequence into one matrix and feed it to the decoder to generate the outputs. Encoder and decoder have different weight matrices.
- Summary of RNN types:   
   ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//12_different_types_of_rnn.jpg)
- There is another architecture which is the **attention** architecture which we will talk about in chapter 3.

### Language model and sequence generation
- RNNs do very well in language model problems. In this section, we will build a language model using RNNs.
- **What is a language model**
  - Let's say we are solving a speech recognition problem and someone says a sentence that can be interpreted into to two sentences:
    - The apple and **pair** salad
    - The apple and **pear** salad
  - **Pair** and **pear** sounds exactly the same, so how would a speech recognition application choose from the two.
  - That's where the language model comes in. It gives a probability for the two sentences and the application decides the best based on this probability.
- The job of a language model is to give a probability of any given sequence of words.
- **How to build language models with RNNs?**
  - The first thing is to get a **training set**: a large corpus of target language text.
  - Then tokenize this training set by getting the vocabulary and then one-hot each word.
  - Put an end of sentence token `<EOS>` with the vocabulary and include it with each converted sentence. Also, use the token `<UNK>` for the unknown words.
- Given the sentence "Cats average 15 hours of sleep a day. `<EOS>`"
  - In training time we will use this:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//13.png)
  - The loss function is defined by cross-entropy loss:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//14.png)
    - `i`  is for all elements in the corpus, `t` - for all timesteps.
- To use this model:
  1.  For predicting the chance of **next word**, we feed the sentence to the RNN and then get the final y<sup>^\<t></sup> hot vector and sort it by maximum probability.
  2.  For taking the **probability of a sentence**, we compute this:
      - p(y<sup><1></sup>, y<sup><2></sup>, y<sup><3></sup>) = p(y<sup><1></sup>) * p(y<sup><2></sup> | y<sup><1></sup>) * p(y<sup><3></sup> | y<sup><1></sup>, y<sup><2></sup>)
      - This is simply feeding the sentence into the RNN and multiplying the probabilities (outputs).

### Sampling novel sequences
- After a sequence model is trained on a language model, to check what the model has learned you can apply it to sample novel sequence.
- Lets see the steps of how we can sample a novel sequence from a trained sequence language model:
  1. Given this model:   
     ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//15.png)
  2. We first pass a<sup><0></sup> = zeros vector, and x<sup><1></sup> = zeros vector.
  3. Then we choose a prediction randomly from distribution obtained by y&#770;<sup><1></sup>. For example it could be "The".
     - In numpy this can be implemented using: `numpy.random.choice(...)`
     - This is the line where you get a random beginning of the sentence each time you sample run a novel sequence.
  4. We pass the last predicted word with the calculated  a<sup><1></sup>
  5. We keep doing 3 & 4 steps for a fixed length or until we get the `<EOS>` token.
  6. You can reject any `<UNK>` token if you mind finding it in your output.
- So far we have to build a word-level language model. It's also possible to implement a **character-level** language model.
- In the character-level language model, the vocabulary will contain `[a-zA-Z0-9]`, punctuation, special characters and possibly <EOS> token.
- Character-level language model has some pros and cons compared to the word-level language model
  - Pros:
    1. There will be no `<UNK>` token - it can create any word.
  - Cons:
    1. The main disadvantage is that you end up with much longer sequences. 
    2. Character-level language models are not as good as word-level language models at capturing long range dependencies between how the the earlier parts of the sentence also affect the later part of the sentence.
    3. Also more computationally expensive and harder to train.
- The trend Andrew has seen in NLP is that for the most part, a word-level language model is still used, but as computers get faster there are more and more applications where people are, at least in some special cases, starting to look at more character-level models. Also, they are used in specialized applications where you might need to deal with unknown words or other vocabulary words a lot. Or they are also used in more specialized applications where you have a more specialized vocabulary.

### Vanishing gradients with RNNs
- One of the problems with naive RNNs that they run into **vanishing gradient** problem.

- An RNN that process a sequence data with the size of 10,000 time steps, has 10,000 deep layers which is very hard to optimize.

- Let's take an example. Suppose we are working with language modeling problem and there are two sequences that model tries to learn:

  - "The **cat**, which already ate ..., **was** full"
  - "The **cats**, which already ate ..., **were** full"
  - Dots represent many words in between.

- What we need to learn here that "was" came with "cat" and that "were" came with "cats". The naive RNN is not very good at capturing very long-term dependencies like this.

- As we have discussed in Deep neural networks, deeper networks are getting into the vanishing gradient problem. That also happens with RNNs with a long sequence size.   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//16.png)   
  - For computing the word "was", we need to compute the gradient for everything behind. Multiplying fractions tends to vanish the gradient, while multiplication of large number tends to explode it.
  - Therefore some of your weights may not be updated properly.

- In the problem we descried it means that its hard for the network to memorize "was" word all over back to "cat". So in this case, the network won't identify the singular/plural words so that it gives it the right grammar form of verb was/were.

- The conclusion is that RNNs aren't good in **long-term dependencies**.

- > In theory, RNNs are absolutely capable of handling such “long-term dependencies.” A human could carefully pick parameters for them to solve toy problems of this form. Sadly, in practice, RNNs don’t seem to be able to learn them. http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- _Vanishing gradients_ problem tends to be the bigger problem with RNNs than the _exploding gradients_ problem. We will discuss how to solve it in next sections.

- Exploding gradients can be easily seen when your weight values become `NaN`. So one of the ways solve exploding gradient is to apply **gradient clipping** means if your gradient is more than some threshold - re-scale some of your gradient vector so that is not too big. So there are cliped according to some maximum value.

  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//26.png)

- **Extra**:
  - Solutions for the Exploding gradient problem:
    - Truncated backpropagation.
      - Not to update all the weights in the way back.
      - Not optimal. You won't update all the weights.
    - Gradient clipping.
  - Solution for the Vanishing gradient problem:
    - Weight initialization.
      - Like He initialization.
    - Echo state networks.
    - Use LSTM/GRU networks.
      - Most popular.
      - We will discuss it next.

### Gated Recurrent Unit (GRU)
- GRU is an RNN type that can help solve the vanishing gradient problem and can remember the long-term dependencies.

- The basic RNN unit can be visualized to be like this:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//17.png)

- We will represent the GRU with a similar drawings.

- Each layer in **GRUs**  has a new variable `C` which is the memory cell. It can tell to whether memorize something or not.

- In GRUs, C<sup>\<t></sup> = a<sup>\<t></sup>

- Equations of the GRUs:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//18.png)
  - The update gate is between 0 and 1
    - To understand GRUs imagine that the update gate is either 0 or 1 most of the time.
  - So we update the memory cell based on the update cell and the previous cell.

- Lets take the cat sentence example and apply it to understand this equations:

  - Sentence: "The **cat**, which already ate ........................, **was** full"

  - We will suppose that U is 0 or 1 and is a bit that tells us if a singular word needs to be memorized.

  - Splitting the words and get values of C and U at each place:

    - | Word    | Update gate(U)             | Cell memory (C) |
      | ------- | -------------------------- | --------------- |
      | The     | 0                          | val             |
      | cat     | 1                          | new_val         |
      | which   | 0                          | new_val         |
      | already | 0                          | new_val         |
      | ...     | 0                          | new_val         |
      | was     | 1 (I don't need it anymore)| newer_val       |
      | full    | ..                         | ..              |
- Drawing for the GRUs   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//19.png)
  
  - Drawings like in http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is so popular and makes it easier to understand GRUs and LSTMs. But Andrew Ng finds it's better to look at the equations.
- Because the update gate U is usually a small number like 0.00001, GRUs doesn't suffer the vanishing gradient problem.
  
  - In the equation this makes C<sup>\<t></sup> = C<sup>\<t-1></sup> in a lot of cases.
- Shapes:
  - a<sup>\<t></sup> shape is (NoOfHiddenNeurons, 1)
  - c<sup>\<t></sup> is the same as a<sup>\<t></sup>
  - c<sup>~\<t></sup> is the same as a<sup>\<t></sup>
  - u<sup>\<t></sup> is also the same dimensions of a<sup>\<t></sup>
- The multiplication in the equations are element wise multiplication.
- What has been descried so far is the Simplified GRU unit. Let's now describe the full one:
  - The full GRU contains a new gate that is used with to calculate the candidate C. The gate tells you how relevant is C<sup>\<t-1></sup> to C<sup>\<t></sup>
  - Equations:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//20.png)
  - Shapes are the same
- So why we use these architectures, why don't we change them, how we know they will work, why not add another gate, why not use the simpler GRU instead of the full GRU; well researchers has experimented over years all the various types of these architectures with many many different versions and also addressing the vanishing gradient problem. They have found that full GRUs are one of the best RNN architectures  to be used for many different problems. You can make your design but put in mind that GRUs and LSTMs are standards.

### Long Short Term Memory (LSTM)
- LSTM - the other type of RNN that can enable you to account for long-term dependencies. It's more powerful and general than GRU.
- In LSTM , C<sup>\<t></sup> != a<sup>\<t></sup>
- Here are the equations of an LSTM unit:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//21.png)
- In GRU we have an update gate `U`, a relevance gate `r`, and a candidate cell variables C<sup>\~\<t></sup> while in LSTM we have an update gate `U` (sometimes it's called input gate I), a forget gate `F`, an output gate `O`, and a candidate cell variables C<sup>\~\<t></sup>
- Drawings (inspired by http://colah.github.io/posts/2015-08-Understanding-LSTMs/):    
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//22.png)
- Some variants on LSTM includes:
  - LSTM with **peephole connections**.
    - The normal LSTM with C<sup>\<t-1></sup> included with every gate.
- There isn't a universal superior between LSTM and it's variants. One of the advantages of GRU is that it's simpler and can be used to build much bigger network but the LSTM is more powerful and general.

### Bidirectional RNN
- There are still some ideas to let you build much more powerful sequence models. One of them is bidirectional RNNs and another is Deep RNNs.
- As we saw before, here is an example of the Name entity recognition task:  
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//23.png)
- The name **Teddy** cannot be learned from **He** and **said**, but can be learned from **bears**.
- BiRNNs fixes this issue.
- Here is BRNNs architecture:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//24.png)
- Note, that BiRNN is an **acyclic graph**.
- Part of the forward propagation goes from left to right, and part - from right to left. It learns from both sides.
- To make predictions we use y&#770;<sup>\<t></sup> by using the two activations that come from left and right.
- The blocks here can be any RNN block including the basic RNNs, LSTMs, or GRUs.
- For a lot of NLP or text processing problems, a BiRNN with LSTM appears to be commonly used.
- The disadvantage of BiRNNs that you need the entire sequence before you can process it. For example, in live speech recognition if you use BiRNNs you will need to wait for the person who speaks to stop to take the entire sequence and then make your predictions.

### Deep RNNs
- In a lot of cases the standard one layer RNNs will solve your problem. But in some problems its useful to stack some RNN layers to make a deeper network.
- For example, a deep RNN with 3 layers would look like this:  
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//25.png)
- In feed-forward deep nets, there could be 100 or even 200 layers. In deep RNNs stacking 3 layers is already considered deep and expensive to train.
- In some cases you might see some feed-forward network layers connected after recurrent cell.


### Back propagation with RNNs
- > In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes care of the backward pass, so most deep learning engineers do not need to bother with the details of the backward pass. If however you are an expert in calculus and want to see the details of backprop in RNNs, you can work through this optional portion of the notebook.

- The quote is taken from this [notebook](https://www.coursera.org/learn/nlp-sequence-models/notebook/X20PE/building-a-recurrent-neural-network-step-by-step). If you want the details of the back propagation with programming notes look at the linked notebook.

## Sequence models & Attention mechanism

> Sequence models can be augmented using an attention mechanism. This algorithm will help your model understand where it should focus its attention given a sequence of inputs. This week, you will also learn about speech recognition and how to deal with audio data.

### Various sequence to sequence architectures

#### BLEU Score
- One of the challenges of machine translation, is that given a sentence in a language there are one or more possible good translation in another language. So how do we evaluate our results?
- The way we do this is by using **BLEU score**. BLEU stands for _bilingual evaluation understudy_.
- The intuition is: as long as the machine-generated translation is pretty close to any of the references provided by humans, then it will get a high BLEU score.
- Let's take an example:
  - X = "Le chat est sur le tapis."
  - Y1 = "The cat is on the mat." (human reference 1)
  - Y2 = "There is a cat on the mat." (human reference 2)
  - Suppose that the machine outputs: "the the the the the the the."
  - One way to evaluate the machine output is to look at each word in the output and check if it is in the references. This is called _precision_:
    - precision = 7/7  because "the" appeared in Y1 or Y2
  - This is not a useful measure!
  - We can use a modified precision in which we are looking for the reference with the maximum number of a particular word and set the maximum appearing of this word to this number. So:
    - modified precision = 2/7 because the max is 2 in Y1
    - We clipped the 7 times by the max which is 2.
  - Here we are looking at one word at a time - unigrams, we may look at n-grams too
- BLEU score on bigrams
  - The **n-grams** typically are collected from a text or speech corpus. When the items are words, **n-grams** may also be called shingles. An **n-gram** of size 1 is referred to as a "unigram"; size 2 is a "bigram" (or, less commonly, a "digram"); size 3 is a "trigram".
  - X = "Le chat est sur le tapis."
  - Y1 = "The cat is on the mat."
  - Y2 = "There is a cat on the mat."
  - Suppose that the machine outputs: "the cat the cat on the mat."
  - The bigrams in the machine output:
  
    | Pairs      | Count | Count clip |
    | ---------- | ----- | ---------- |
    | the cat    | 2     | 1 (Y1)     |
    | cat the    | 1     | 0          |
    | cat on     | 1     | 1 (Y2)     |
    | on the     | 1     | 1 (Y1)     |
    | the mat    | 1     | 1 (Y1)     |
    | **Totals** | 6     | 4          |

    Modified precision = sum(Count clip) / sum(Count) = 4/6
- So here are the equations for modified presicion for the n-grams case:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//60.png)
- Let's put this together to formalize the BLEU score:
  - **P<sub>n</sub>** = Bleu score on one type of n-gram
  - **Combined BLEU score** = BP * exp(1/n * sum(P<sub>n</sub>))
    - For example if we want BLEU for 4, we compute P<sub>1</sub>, P<sub>2</sub>, P<sub>3</sub>, P<sub>4</sub> and then average them and take the exp.
  - **BP** is called **BP penalty** which stands for brevity penalty. It turns out that if a machine outputs a small number of words it will get a better score so we need to handle that.   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//62.png)
- BLEU score has several open source implementations. 
- It is used in a variety of systems like machine translation and image captioning.

#### Attention Model Intuition
- So far we were using sequence to sequence models with an encoder and decoders. There is a technique called _attention_ which makes these models even better.
- The attention idea has been one of the most influential ideas in deep learning. 
- The problem of long sequences:
  - Given this model, inputs, and outputs.   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//63.png)
  - The encoder should memorize this long sequence into one vector, and the decoder has to process this vector to generate the translation.
  - If a human would translate this sentence, he/she wouldn't read the whole sentence and memorize it then try to translate it. He/she translates a part at a time.
  - The performance of this model decreases if a sentence is long.
  - We will discuss the attention model that works like a human that looks at parts at a time. That will significantly increase the accuracy even with longer sequence:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//64.png)
    -  Blue is the normal model, while green is the model with attention mechanism.
- In this section we will give just some intuitions about the attention model and in the next section we will discuss it's details.
- At first the attention model was developed for machine translation but then other applications used it like computer vision and new architectures like Neural Turing machine.
- The attention model was descried in this paper:
  - [Bahdanau et. al., 2014. Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
- Now for the intuition:
  - Suppose that our encoder is a bidirectional RNN:
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//65.png)
  - We give the French sentence to the encoder and it should generate a vector that represents the inputs.
  - Now to generate the first word in English which is "Jane" we will make another RNN which is the decoder.
  - Attention weights are used to specify which words are needed when to generate a word. So to generate "jane" we will look at "jane", "visite", "l'Afrique"   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//66.png)
  - alpha<sup>\<1,1></sup>, alpha<sup>\<1,2></sup>, and alpha<sup>\<1,3></sup> are the attention weights being used.
  - And so to generate any word there will be a set of attention weights that controls which words we are looking at right now.   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//67.png)

#### Attention Model
- Lets formalize the intuition from the last section into the exact details on how this can be implemented.
- First we will have an bidirectional RNN (most common is LSTMs) that encodes French language:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//68.png)
- For learning purposes, lets assume that a<sup>\<t'></sup> will include the both directions activations at time step t'.
- We will have a unidirectional RNN to produce the output using a context `c` which is computed using the attention weights, which denote how much information does the output needs to look in a<sup>\<t'></sup>   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//69.png)
- Sum of the attention weights for each element in the sequence should be 1:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//70.png)
- The context `c` is calculated using this equation:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//71.png)
- Lets see how can we compute the attention weights:
  - So alpha<sup>\<t, t'></sup> = amount of attention y<sup>\<t></sup> should pay to a<sup>\<t'></sup>
    - Like for example we payed attention to the first three words through alpha<sup>\<1,1></sup>, alpha<sup>\<1,2></sup>, alpha<sup>\<1,3></sup>
  - We are going to softmax the attention weights so that their sum is 1:   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//72.png)
  - Now we need to know how to calculate e<sup>\<t, t'></sup>. We will compute e using a small neural network (usually 1-layer, because we will need to compute this a lot):   
    ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//73.png)
    - s<sup>\<t-1></sup> is the hidden state of the RNN s, and a<sup>\<t'></sup> is the activation of the other bidirectional RNN. 
- One of the disadvantages of this algorithm is that it takes quadratic time or quadratic cost to run.
- One fun way to see how attention works is by visualizing the attention weights:   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//74.png)

## Extras

### Machine translation attention model (from notebooks)

- The model is built with keras layers.
- The attention model.   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//83.png)
  - There are two separate LSTMs in this model. Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through $T_x$ time steps; the post-attention LSTM goes through $T_y$ time steps. 
  - The post-attention LSTM passes $s^{\langle t \rangle}, c^{\langle t \rangle}$ from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations $s^{\langle t\rangle}$. But since we are using an LSTM here, the LSTM has both the output activation $s^{\langle t\rangle}$ and the hidden cell state $c^{\langle t\rangle}$. However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time $t$ does will not take the specific generated $y^{\langle t-1 \rangle}$ as input; it only takes $s^{\langle t\rangle}$ and $c^{\langle t\rangle}$ as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date. 
- What one "Attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$, which are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$).   
  ![](https://raw.githubusercontent.com/ashishpatel26/DeepLearning.ai-Summary/master/5-%20Sequence%20Models/Images//84.png)
  - The diagram uses a `RepeatVector` node to copy $s^{\langle t-1 \rangle}$'s value $T_x$ times, and then `Concatenation` to concatenate $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$ to compute $e^{\langle t, t'}$, which is then passed through a softmax to compute $\alpha^{\langle t, t' \rangle}$. 
