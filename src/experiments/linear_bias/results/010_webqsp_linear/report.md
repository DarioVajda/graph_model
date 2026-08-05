# Sweep report: 010_webqsp_linear

21 run(s) recorded.

Shared across all runs:

- `mode` = train
- `sweep_id` = 010_webqsp_linear
- `train_datasets` = [webqsp]
- `eval_datasets` = [webqsp]
- `selection_dataset` = 
- `model_name` = meta-llama/Llama-3.2-1B
- `prompt_style` = plain
- `lora_r` = 64
- `k_hop` = 0
- `k_hop_directed` = False
- `graph_attn_impl` = flex
- `dtype` = bf16
- `lr` = 0.0001
- `bias_lr` = 0.0050
- `num_epochs` = 15
- `batch_size` = 2
- `accumulation_steps` = 4
- `max_steps` = -1
- `boundary_loss_weight` = 1.0000
- `bias_weight_decay` = 0.0000
- `lora_dropout` = 0.1500
- `rel_mode` = last_1
- `max_nodes` = 512
- `n_max` = 50
- `versions` = 8
- `magnetic_m` = 128
- `data_seed` = 42
- `magnetic_groups` = 0
- `data_format_version` = 3
- `cvt_collapse` = True
- `question_node` = isolated
- `graph_construction` = levi

| sweep_run | run_name | seed | eval_sel_f1 | eval_webqsp_f1 | test_webqsp_f1 | eval_webqsp_hits1 | test_webqsp_hits1 | eval_webqsp_hit_star | test_webqsp_hit_star | eval_webqsp_em_accuracy | test_webqsp_em_accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0000_seed0_magneticTrue_magnetic_linearFalse_magnetic_m_collate128 | 010_webqsp_linear_0000_seed0_magneticTrue_magnetic_linearFalse_magnetic_m_collate128 | 0 | 0.7322 | 0.7322 | 0.7190 | 0.7846 | 0.7789 | 0.8293 | 0.8335 | 0.3659 | 0.3686 |
| 0001_seed0_magneticTrue_magnetic_linearFalse_magnetic_m_collate16 | 010_webqsp_linear_0001_seed0_magneticTrue_magnetic_linearFalse_magnetic_m_collate16 | 0 | 0.7103 | 0.7103 | 0.6968 | 0.7846 | 0.7598 | 0.8333 | 0.8133 | 0.3455 | 0.3550 |
| 0002_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate128 | 010_webqsp_linear_0002_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate128 | 0 | 0.6856 | 0.6856 | 0.6699 | 0.7358 | 0.7359 | 0.8089 | 0.7942 | 0.3252 | 0.3305 |
| 0003_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate64 | 010_webqsp_linear_0003_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate64 | 0 | 0.6771 | 0.6771 | 0.6717 | 0.7561 | 0.7383 | 0.8171 | 0.7936 | 0.2927 | 0.3354 |
| 0004_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate32 | 010_webqsp_linear_0004_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate32 | 0 | 0.6909 | 0.6909 | 0.6593 | 0.7642 | 0.7267 | 0.8211 | 0.7948 | 0.3374 | 0.3225 |
| 0005_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate16 | 010_webqsp_linear_0005_seed0_magneticFalse_magnetic_linearTrue_magnetic_m_collate16 | 0 | 0.6563 | 0.6563 | 0.6434 | 0.7073 | 0.7144 | 0.7886 | 0.7887 | 0.3293 | 0.3114 |
| 0006_seed0_magneticFalse_magnetic_linearFalse_magnetic_m_collate0 | 010_webqsp_linear_0006_seed0_magneticFalse_magnetic_linearFalse_magnetic_m_collate0 | 0 | 0.4677 | 0.4677 | 0.4594 | 0.5488 | 0.5338 | 0.6463 | 0.6155 | 0.2073 | 0.2119 |
| 0007_seed1_magneticTrue_magnetic_linearFalse_magnetic_m_collate128 | 010_webqsp_linear_0007_seed1_magneticTrue_magnetic_linearFalse_magnetic_m_collate128 | 1 | 0.7351 | 0.7351 | 0.7129 | 0.8049 | 0.7740 | 0.8374 | 0.8188 | 0.3943 | 0.3864 |
| 0008_seed1_magneticTrue_magnetic_linearFalse_magnetic_m_collate16 | 010_webqsp_linear_0008_seed1_magneticTrue_magnetic_linearFalse_magnetic_m_collate16 | 1 | 0.7165 | 0.7165 | 0.7159 | 0.7764 | 0.7758 | 0.8293 | 0.8274 | 0.3659 | 0.3888 |
| 0009_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate128 | 010_webqsp_linear_0009_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate128 | 1 | 0.6894 | 0.6894 | 0.6687 | 0.7358 | 0.7322 | 0.8211 | 0.7936 | 0.3293 | 0.3317 |
| 0010_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate64 | 010_webqsp_linear_0010_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate64 | 1 | 0.6869 | 0.6869 | 0.6706 | 0.7683 | 0.7242 | 0.8171 | 0.7930 | 0.3333 | 0.3317 |
| 0011_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate32 | 010_webqsp_linear_0011_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate32 | 1 | 0.6723 | 0.6723 | 0.6515 | 0.7276 | 0.7131 | 0.8211 | 0.7795 | 0.3171 | 0.3360 |
| 0012_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate16 | 010_webqsp_linear_0012_seed1_magneticFalse_magnetic_linearTrue_magnetic_m_collate16 | 1 | 0.6453 | 0.6453 | 0.6403 | 0.7114 | 0.7082 | 0.7846 | 0.7918 | 0.2927 | 0.3010 |
| 0013_seed1_magneticFalse_magnetic_linearFalse_magnetic_m_collate0 | 010_webqsp_linear_0013_seed1_magneticFalse_magnetic_linearFalse_magnetic_m_collate0 | 1 | 0.4655 | 0.4655 | 0.4580 | 0.5488 | 0.5319 | 0.6504 | 0.6253 | 0.1992 | 0.2027 |
| 0014_seed2_magneticTrue_magnetic_linearFalse_magnetic_m_collate128 | 010_webqsp_linear_0014_seed2_magneticTrue_magnetic_linearFalse_magnetic_m_collate128 | 2 | 0.7323 | 0.7323 | 0.7268 | 0.8049 | 0.7844 | 0.8293 | 0.8305 | 0.3862 | 0.3857 |
| 0015_seed2_magneticTrue_magnetic_linearFalse_magnetic_m_collate16 | 010_webqsp_linear_0015_seed2_magneticTrue_magnetic_linearFalse_magnetic_m_collate16 | 2 | 0.7404 | 0.7404 | 0.7060 | 0.7724 | 0.7611 | 0.8374 | 0.8176 | 0.3740 | 0.3710 |
| 0016_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate128 | 010_webqsp_linear_0016_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate128 | 2 | 0.6842 | 0.6842 | 0.6740 | 0.7439 | 0.7518 | 0.8089 | 0.7936 | 0.3455 | 0.3415 |
| 0017_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate64 | 010_webqsp_linear_0017_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate64 | 2 | 0.6931 | 0.6931 | 0.6823 | 0.7480 | 0.7445 | 0.8211 | 0.8084 | 0.3252 | 0.3409 |
| 0018_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate32 | 010_webqsp_linear_0018_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate32 | 2 | 0.6701 | 0.6701 | 0.6565 | 0.7195 | 0.7328 | 0.8171 | 0.7862 | 0.3415 | 0.3249 |
| 0019_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate16 | 010_webqsp_linear_0019_seed2_magneticFalse_magnetic_linearTrue_magnetic_m_collate16 | 2 | 0.6662 | 0.6662 | 0.6424 | 0.7317 | 0.7150 | 0.8130 | 0.7856 | 0.3374 | 0.3059 |
| 0020_seed2_magneticFalse_magnetic_linearFalse_magnetic_m_collate0 | 010_webqsp_linear_0020_seed2_magneticFalse_magnetic_linearFalse_magnetic_m_collate0 | 2 | 0.4699 | 0.4699 | 0.4523 | 0.5366 | 0.5387 | 0.6341 | 0.6112 | 0.2317 | 0.2168 |
