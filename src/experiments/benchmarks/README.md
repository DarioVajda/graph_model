## Text-Attributed Graph (TAG) Benchmarks

In this folder, we have the code used for training and evaluating GTLM on the Cora, PubMed, OGBN-Arxiv, and Reddit datasets.

### Instructions for replicating the experiments

Download the raw datasets according to the [RGLM repository](https://github.com/zhongjian-zhang/RGLM). These datasets can be found on Google Drive at [this link](https://drive.google.com/drive/folders/1aPlqxTUjRPUCNlRS-OpaRToEhZb61ffu) (we will also publish the dataset ourselves, but only after the blind submissions are over). These files should be saved in a folder called `raw_data`, inside of this directory.

The data can be processed and saved in our `.gtds` format by running:
```
python3 -m src.experiments.process_data
```
Note that the `process_data.py` file has to be configured inside of its main function. You can change the source dataset, the text attribute construction method, the number of neighborhood samples for each graph, and the maximum number of samples.

After processing the dataset, you will have created a directory called `processed_data/{processed_dataset_name}`

Finally, the model can be trained by running the following command, and changing any of these hyperparameters:
```
python3 -m src.experiments.benchmarks \
    --model_name=meta-llama/Llama-3.2-1B \
    --dataset_name={processed_dataset_name} \
    --num_epochs=10 \
    --lora_r=64 \
    --learning_rate=1e-5 \
    --bias_learning_rate=1e-2 \
    --eval_every=50
```

A full list of tuneable hyperparameters can be explored inside of the `__main__.py` file of this directory.

The exact hyperparameters used to obtain the results from our paper can be found in the Appendix.

The models are evaluated automatically right after finishing training, but you can also manually evaluate some model checkpoint, by running the `test.py` file, with the appropriate settings.