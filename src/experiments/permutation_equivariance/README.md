## Backward Compatibility with RoPE

In this experiment, I tested whether or not the outputs will be the same regardless of the prefix node permutation in the input sequence.

✅ Assumption found to be true


### Run the experiment with the following command:

```
python3 -m src.experiments.permutation_equivariance
```