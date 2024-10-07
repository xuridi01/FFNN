# Experiments with different optimizers and dropout on FFNN

#### Starting params

- epochs = 10
- starting learning rate = 0.5

### - Experiments with optimizers: 

### Mini-batch GD
- Epoch 1/10; Loss: 0.0375
- Epoch 2/10; Loss: 0.0249
- Epoch 3/10; Loss: 0.0200
- Epoch 4/10; Loss: 0.0177
- Epoch 5/10; Loss: 0.0163
- Epoch 6/10; Loss: 0.0153
- Epoch 7/10; Loss: 0.0146
- Epoch 8/10; Loss: 0.0140
- Epoch 9/10; Loss: 0.0136
- Epoch 10/10; Loss: 0.0132
Correct: 9183/10000

### SGD
- Epoch 1/10; Loss: 0.0282
- Epoch 2/10; Loss: 0.0125
- Epoch 3/10; Loss: 0.0094
- Epoch 4/10; Loss: 0.0079
- Epoch 5/10; Loss: 0.0069
- Epoch 6/10; Loss: 0.0062
- Epoch 7/10; Loss: 0.0056
- Epoch 8/10; Loss: 0.0052
- Epoch 9/10; Loss: 0.0049
- Epoch 10/10; Loss: 0.0046
Correct: 9784/10000

### AdaGrad
- Epoch 1/10; Loss: 310.9993
- Epoch 2/10; Loss: 0.1000
- Epoch 3/10; Loss: 0.1000
- Epoch 4/10; Loss: 0.1000
- Epoch 5/10; Loss: 0.1000
- Epoch 6/10; Loss: 0.1000
- Epoch 7/10; Loss: 0.1000
- Epoch 8/10; Loss: 0.1000
- Epoch 9/10; Loss: 0.1000
- Epoch 10/10; Loss: 0.1000
Correct: 980/10000

We can notice that AdaGrad cant handle this learning rate

AdaGrad with learning rate = 0.01:
- Epoch 1/10; Loss: 0.0344
- Epoch 2/10; Loss: 0.0281
- Epoch 3/10; Loss: 0.0268
- Epoch 4/10; Loss: 0.0260
- Epoch 5/10; Loss: 0.0255
- Epoch 6/10; Loss: 0.0251
- Epoch 7/10; Loss: 0.0248
- Epoch 8/10; Loss: 0.0246
- Epoch 9/10; Loss: 0.0243
- Epoch 10/10; Loss: 0.0242
Correct: 7724/10000

### Adam
- Epoch 1/10; Loss: 107.31697190968372
- Epoch 2/10; Loss: 0.10000000149011612
- Epoch 3/10; Loss: 0.10000000149011612
- Epoch 4/10; Loss: 0.10000000149011612
- Epoch 5/10; Loss: 0.10000000149011612
- Epoch 6/10; Loss: 0.10000000149011612
- Epoch 7/10; Loss: 0.10000000149011612
- Epoch 8/10; Loss: 0.10000000149011612
- Epoch 9/10; Loss: 0.10000000149011612
- Epoch 10/10; Loss: 0.10000000149011612
Correct: 980/10000

We can notice that Adam cant handle this learning rate as well

Adam with learning rate = 0.001:
- Epoch 1/10; Loss: 0.0218
- Epoch 2/10; Loss: 0.0153
- Epoch 3/10; Loss: 0.0139
- Epoch 4/10; Loss: 0.0132
- Epoch 5/10; Loss: 0.0127
- Epoch 6/10; Loss: 0.0123
- Epoch 7/10; Loss: 0.0120
- Epoch 8/10; Loss: 0.0117
- Epoch 9/10; Loss: 0.0116
- Epoch 10/10; Loss: 0.0114
Correct: 8851/10000

### - Experiments with Dropout:

### SGD
- Epoch 1/10; Loss: 0.0274
- Epoch 2/10; Loss: 0.0164
- Epoch 3/10; Loss: 0.0135
- Epoch 4/10; Loss: 0.0119
- Epoch 5/10; Loss: 0.0108
- Epoch 6/10; Loss: 0.0100
- Epoch 7/10; Loss: 0.0093
- Epoch 8/10; Loss: 0.0088
- Epoch 9/10; Loss: 0.0085
- Epoch 10/10; Loss: 0.0081
Correct: 9710/10000

### Adam
- Epoch 1/10; Loss: 0.0201
- Epoch 2/10; Loss: 0.0124
- Epoch 3/10; Loss: 0.0104
- Epoch 4/10; Loss: 0.0092
- Epoch 5/10; Loss: 0.0085
- Epoch 6/10; Loss: 0.0080
- Epoch 7/10; Loss: 0.0076
- Epoch 8/10; Loss: 0.0075
- Epoch 9/10; Loss: 0.0071
- Epoch 10/10; Loss: 0.0069
Correct: 9756/10000