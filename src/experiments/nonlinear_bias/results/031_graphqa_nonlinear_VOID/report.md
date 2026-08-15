# Sweep report: 031_graphqa_nonlinear

36 run(s) recorded.

Shared across all runs:

- `mode` = train
- `sweep_id` = 031_graphqa_nonlinear
- `graph_type` = standard
- `question_node` = isolated
- `spd` = False
- `rrwp` = False
- `magnetic` = False
- `magnetic_groups` = 0
- `magnetic_linear` = False
- `magnetic_magnitude` = False
- `magnetic_hybrid` = False
- `magnetic_linear_v2` = False
- `magnetic_nonlinear` = True
- `magnetic_struct_dim` = 64
- `magnetic_m_collate` = 64
- `bias_self_node` = True
- `model_name` = meta-llama/Llama-3.2-1B
- `impl` = v2-eager
- `dtype` = fp32
- `k_hop` = 0
- `k_hop_directed` = False
- `max_spd` = 8
- `max_rw_steps` = 16
- `magnetic_dim` = 64
- `magnetic_q` = 0.2500
- `magnetic_m` = 0
- `lora` = True
- `lora_r` = 16
- `lora_alpha` = 32
- `lora_dropout` = 0.0500
- `lr` = 3e-05
- `num_epochs` = 20
- `batch_size` = 4
- `accumulation_steps` = 8
- `eval_steps` = 20
- `max_steps` = -1
- `max_length` = 1024
- `val_source` = official
- `train_size` = 1000
- `val_size` = 500
- `test_size` = 500

| sweep_run | run_name | task | arm | magnetic_pool | seed | bias_lr | test_accuracy | best_val_accuracy | test_loss | train_runtime_s | train_steps_per_second |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0000_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0000_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.005 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0050 | 0.9820 | 0.9780 | 0.0174 | 1012.9017 | 0.6120 |
| 0001_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0001_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.005 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0050 | 0.0900 | 0.1140 | 0.6274 | 914.3613 | 0.6780 |
| 0002_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0002_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.001 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0010 | 0.7280 | 0.7260 | 0.2013 | 866.9538 | 0.7150 |
| 0003_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0003_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.001 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0010 | 0.0900 | 0.1140 | 0.6274 | 918.2348 | 0.6750 |
| 0004_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0004_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.005 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0050 | 0.0800 | 0.1160 | 0.6538 | 1012.2435 | 0.6130 |
| 0005_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0005_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.005 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0050 | 0.0800 | 0.1160 | 0.6538 | 935.6096 | 0.6630 |
| 0006_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0006_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.001 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0010 | 0.5480 | 0.5880 | 0.2847 | 1083.0524 | 0.5720 |
| 0007_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0007_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.001 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0010 | 0.0800 | 0.1160 | 0.6538 | 958.0331 | 0.6470 |
| 0008_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0008_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.005 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0050 | 0.0960 | 0.1120 | 0.6277 | 1087.5488 | 0.5700 |
| 0009_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0009_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.005 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0050 | 0.0960 | 0.1120 | 0.6277 | 922.4457 | 0.6720 |
| 0010_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0010_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.001 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0010 | 0.0960 | 0.1120 | 0.6277 | 1049.4979 | 0.5910 |
| 0011_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0011_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.001 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0010 | 0.0960 | 0.1120 | 0.6277 | 943.4266 | 0.6570 |
| 0012_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0012_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.005 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0050 | 0.4700 | 0.5260 | 0.2362 | 1049.3151 | 0.5910 |
| 0013_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0013_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.005 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0050 | 0.6540 | 0.7380 | 0.1559 | 784.7144 | 0.7900 |
| 0014_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0014_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.001 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0010 | 0.4700 | 0.5260 | 0.2362 | 1015.5251 | 0.6110 |
| 0015_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0015_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.001 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0010 | 0.6380 | 0.7280 | 0.1561 | 958.8949 | 0.6470 |
| 0016_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0016_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.005 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0050 | 0.4700 | 0.5260 | 0.2363 | 1024.5332 | 0.6050 |
| 0017_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0017_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.005 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0050 | 0.4700 | 0.5260 | 0.2363 | 778.1636 | 0.7970 |
| 0018_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0018_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.001 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0010 | 0.4700 | 0.5280 | 0.2363 | 1649.1049 | 0.3760 |
| 0019_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0019_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.001 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0010 | 0.4700 | 0.5260 | 0.2363 | 986.9640 | 0.6280 |
| 0020_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0020_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.005 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0050 | 0.4680 | 0.5260 | 0.2388 | 1016.8162 | 0.6100 |
| 0021_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0021_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.005 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0050 | 0.4680 | 0.5260 | 0.2388 | 981.7410 | 0.6320 |
| 0022_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0022_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.001 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0010 | 0.4680 | 0.5260 | 0.2388 | 1144.1615 | 0.5420 |
| 0023_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0023_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.001 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0010 | 0.4680 | 0.5260 | 0.2388 | 800.4204 | 0.7750 |
| 0024_taskedge_count_seed42_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0024_taskedge_count_seed42_magnetic_poolattn_bias_lr0.005 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0050 | 0.0240 | 0.0400 | 0.8546 | 912.2827 | 0.6800 |
| 0025_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0025_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.005 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0050 | 0.0220 | 0.0400 | 0.8652 | 955.3504 | 0.6490 |
| 0026_taskedge_count_seed42_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0026_taskedge_count_seed42_magnetic_poolattn_bias_lr0.001 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0010 | 0.0000 | 0.0000 | nan | 1050.3703 | 0.5900 |
| 0027_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0027_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.001 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0010 | 0.0240 | 0.0400 | 0.8546 | 912.4100 | 0.6800 |
| 0028_taskedge_count_seed43_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0028_taskedge_count_seed43_magnetic_poolattn_bias_lr0.005 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0050 | 0.0220 | 0.0400 | 0.8652 | 1081.6420 | 0.5730 |
| 0029_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0029_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.005 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0050 | 0.0220 | 0.0400 | 0.8652 | 693.2451 | 0.8940 |
| 0030_taskedge_count_seed43_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0030_taskedge_count_seed43_magnetic_poolattn_bias_lr0.001 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0010 | 0.0220 | 0.0380 | 0.8935 | 1145.0308 | 0.5410 |
| 0031_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0031_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.001 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0010 | 0.0220 | 0.0400 | 0.8652 | 718.6119 | 0.8630 |
| 0032_taskedge_count_seed44_magnetic_poolattn_bias_lr0.005 | 031_graphqa_nonlinear_0032_taskedge_count_seed44_magnetic_poolattn_bias_lr0.005 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0050 | 0.0260 | 0.0380 | 0.8683 | 1060.0322 | 0.5850 |
| 0033_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.005 | 031_graphqa_nonlinear_0033_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.005 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0050 | 0.4720 | 0.4940 | 0.3309 | 951.1743 | 0.6520 |
| 0034_taskedge_count_seed44_magnetic_poolattn_bias_lr0.001 | 031_graphqa_nonlinear_0034_taskedge_count_seed44_magnetic_poolattn_bias_lr0.001 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0010 | 0.1020 | 0.0900 | 0.6584 | 1644.6052 | 0.3770 |
| 0035_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.001 | 031_graphqa_nonlinear_0035_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.001 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0010 | 0.0260 | 0.0380 | 0.8683 | 787.6445 | 0.7870 |
