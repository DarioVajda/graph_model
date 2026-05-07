## Our tests — Family Tree and Knowledge Graph QA

This directory contains the scripts for generating the synthetic datasets, and for training and evaluating the GTLM model and the LLM baselines.

### Family Tree dataset preparation:
```
python3 -m src.experiments.our_tests.family_tree_prep
```

### Knowledge Graph QA dataset preparation
We can generate three dataset variants with these three python modules:
1. `data_prep.py` – used for generating the `.gtds` format datasets for GTLM.
2. `data_prep_llm.py` – used for generating the dataset in json files, which are used for training the LLM baseline
3. `data_prep_llaga.py` – used for genreating the dataset in the *LLaGA* format, used by the RGLM models.

### Training GTLM
```
python3 -m src.experiments.our_tests \
  --dataset_name={kg_qa/family} \
  --model_name=meta-llama/Llama-3.2-1B \
  --lora_r=32 \
  --batch_size=4 \
  --accumulation_steps=4 \
  --learning_rate=1e-4 \
  --bias_learning_rate=1e-2 \
  --num_epochs=10
```

### Training the LLM baseline
```
python3 -m src.experiments.our_tests.train_llm_baseline
```

### Training the RGLM-Decoder baseline
We relied on the pipeline provided in the [RGLM repository](https://github.com/zhongjian-zhang/RGLM).