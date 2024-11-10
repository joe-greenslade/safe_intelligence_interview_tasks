# Solution and workflow

Intital thoughts after reading brief:
    * Task 1: Shouldn't take long, the use of the multiprocessing library will be needed
    * Task 2: Actually had to perform backward propagation by hand during an exam in May
    * Task 3: Unfamiliar with `Interval Bound Propagation (IBP)`, might need to research before I can complete the task

## Task 1:
* Recieved a Type Error when passing arguments to `pd.read_csv()` in pool.starmap(), will create a util function to handle the arguments
* Managed to get a working solution, output comapred to SingleProccessDataset:

```
Running main.py...
Loading data using single process...
Dataset loading completed in 0.01 seconds
Loading data using multi process...
Dataset loading completed in 4.25 seconds
```

## Task 2:
* Encountered when trying to test NN
```
TypeError: SimpleNeuralNetwork.__init__() got an unexpected keyword argument 'batch_size'
```
Fixed by removing from SimpleNeuralNetwork `batch_size=32,` class initaliser as it's also passed when calling training on the network

* Commited initial solution to Task 2, however, it's not final as I am receiving this error:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x1 and 32x1)
```

for this line:
```
dZ2 = torch.matmul(self.W3.T, dZ3)
```

* A soultion has been found after playing around and tweaking matrix shapes for multiplication. However, for both Epoch's the average loss is nan and the final accurracy is 3.83%:
```
Epoch 1 average loss: nan
...
Epoch 2 average loss: nan
...
Training samples: 8000
Test samples: 2000
Final accuracy: 3.83%
```
I don't believe this to be the final solution due to the average loss not being a number.

* I have tweaked the learning rate to `0.0001` and now achieve this output:
```
Epoch 1 average loss: 0.4588
...
Epoch 2 average loss: 0.0349
...
Training samples: 8000
Test samples: 2000
Final accuracy: 96.17%
```
My Conclusion: The learning rate was too high/large, preventing any actual meaningful gradient descent occuring


## Task 3:
* Unattempted witin time constraints of the technical as current time is 70 mins, will need more time to research `IntervalBoundPropagation` and come up with a solution.
* Will hopefully have a solution by the next interview
