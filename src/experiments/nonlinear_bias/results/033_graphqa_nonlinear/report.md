# Sweep report: 033_graphqa_nonlinear

36 run(s) recorded.

Shared across all runs:

- `mode` = train
- `sweep_id` = 033_graphqa_nonlinear
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
| 0000_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0000_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.005 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0050 | 0.9900 | 0.9900 | 0.0114 | 1041.6714 | 0.5950 |
| 0001_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0001_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.005 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0050 | 0.9920 | 0.9900 | 0.0121 | 912.3378 | 0.6800 |
| 0002_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0002_tasknode_degree_seed42_magnetic_poolattn_bias_lr0.001 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0010 | 0.9840 | 0.9780 | 0.0183 | 1018.3272 | 0.6090 |
| 0003_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0003_tasknode_degree_seed42_magnetic_pooluniform_bias_lr0.001 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0010 | 0.9840 | 0.9700 | 0.0218 | 1094.9578 | 0.5660 |
| 0004_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0004_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.005 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0050 | 0.9840 | 0.9840 | 0.0168 | 845.0141 | 0.7340 |
| 0005_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0005_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.005 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0050 | 0.9860 | 0.9920 | 0.0156 | 593.3601 | 1.0450 |
| 0006_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0006_tasknode_degree_seed43_magnetic_poolattn_bias_lr0.001 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0010 | 0.9880 | 0.9820 | 0.0149 | 1050.4663 | 0.5900 |
| 0007_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0007_tasknode_degree_seed43_magnetic_pooluniform_bias_lr0.001 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0010 | 0.9900 | 0.9880 | 0.0132 | 922.4077 | 0.6720 |
| 0008_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0008_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.005 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0050 | 0.9840 | 0.9820 | 0.0146 | 1048.2988 | 0.5910 |
| 0009_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0009_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.005 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0050 | 1.0000 | 0.9940 | 0.0078 | 715.6974 | 0.8660 |
| 0010_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0010_tasknode_degree_seed44_magnetic_poolattn_bias_lr0.001 | node_degree | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0010 | 0.9980 | 0.9980 | 0.0125 | 877.7476 | 0.7060 |
| 0011_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0011_tasknode_degree_seed44_magnetic_pooluniform_bias_lr0.001 | node_degree | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0010 | 0.9880 | 0.9760 | 0.0194 | 911.2855 | 0.6800 |
| 0012_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0012_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.005 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0050 | 0.8020 | 0.8440 | 0.1241 | 1014.1759 | 0.6110 |
| 0013_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0013_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.005 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0050 | 0.7860 | 0.8380 | 0.1667 | 921.6193 | 0.6730 |
| 0014_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0014_taskshortest_path_seed42_magnetic_poolattn_bias_lr0.001 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0010 | 0.6860 | 0.7340 | 0.1438 | 658.1116 | 0.9420 |
| 0015_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0015_taskshortest_path_seed42_magnetic_pooluniform_bias_lr0.001 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0010 | 0.6700 | 0.7300 | 0.1439 | 935.2824 | 0.6630 |
| 0016_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0016_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.005 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0050 | 0.8260 | 0.8740 | 0.0982 | 650.5154 | 0.9530 |
| 0017_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0017_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.005 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0050 | 0.6800 | 0.7620 | 0.1871 | 917.6237 | 0.6760 |
| 0018_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0018_taskshortest_path_seed43_magnetic_poolattn_bias_lr0.001 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0010 | 0.6580 | 0.7460 | 0.1509 | 1016.3694 | 0.6100 |
| 0019_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0019_taskshortest_path_seed43_magnetic_pooluniform_bias_lr0.001 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0010 | 0.6840 | 0.7480 | 0.1465 | 918.7387 | 0.6750 |
| 0020_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0020_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.005 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0050 | 0.8120 | 0.8600 | 0.1127 | 856.4720 | 0.7240 |
| 0021_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0021_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.005 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0050 | 0.7840 | 0.8460 | 0.1205 | 922.2359 | 0.6720 |
| 0022_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0022_taskshortest_path_seed44_magnetic_poolattn_bias_lr0.001 | shortest_path | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0010 | 0.6840 | 0.7500 | 0.1399 | 1035.8776 | 0.5990 |
| 0023_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0023_taskshortest_path_seed44_magnetic_pooluniform_bias_lr0.001 | shortest_path | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0010 | 0.6780 | 0.7500 | 0.1396 | 946.8735 | 0.6550 |
| 0024_taskedge_count_seed42_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0024_taskedge_count_seed42_magnetic_poolattn_bias_lr0.005 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0050 | 0.4480 | 0.4480 | 0.3302 | 1048.0530 | 0.5920 |
| 0025_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0025_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.005 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0050 | 0.5060 | 0.4900 | 0.3166 | 761.0844 | 0.8150 |
| 0026_taskedge_count_seed42_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0026_taskedge_count_seed42_magnetic_poolattn_bias_lr0.001 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 42 | 0.0010 | 0.4260 | 0.4480 | 0.3425 | 636.6093 | 0.9740 |
| 0027_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0027_taskedge_count_seed42_magnetic_pooluniform_bias_lr0.001 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 42 | 0.0010 | 0.3660 | 0.3960 | 0.3791 | 761.6713 | 0.8140 |
| 0028_taskedge_count_seed43_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0028_taskedge_count_seed43_magnetic_poolattn_bias_lr0.005 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0050 | 0.4640 | 0.4700 | 0.3131 | 1021.9845 | 0.6070 |
| 0029_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0029_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.005 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0050 | 0.4580 | 0.4680 | 0.3321 | 919.2934 | 0.6740 |
| 0030_taskedge_count_seed43_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0030_taskedge_count_seed43_magnetic_poolattn_bias_lr0.001 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 43 | 0.0010 | 0.3720 | 0.3860 | 0.3763 | 1024.5913 | 0.6050 |
| 0031_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0031_taskedge_count_seed43_magnetic_pooluniform_bias_lr0.001 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 43 | 0.0010 | 0.3500 | 0.3980 | 0.3717 | 913.0557 | 0.6790 |
| 0032_taskedge_count_seed44_magnetic_poolattn_bias_lr0.005 | 033_graphqa_nonlinear_0032_taskedge_count_seed44_magnetic_poolattn_bias_lr0.005 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0050 | 0.4780 | 0.4720 | 0.3313 | 1089.7781 | 0.5690 |
| 0033_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.005 | 033_graphqa_nonlinear_0033_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.005 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0050 | 0.4540 | 0.4560 | 0.3321 | 917.4630 | 0.6760 |
| 0034_taskedge_count_seed44_magnetic_poolattn_bias_lr0.001 | 033_graphqa_nonlinear_0034_taskedge_count_seed44_magnetic_poolattn_bias_lr0.001 | edge_count | mag-nonlinear+no-spd+rrwp+selfnode | attn | 44 | 0.0010 | 0.3740 | 0.4040 | 0.3643 | 1109.1653 | 0.5590 |
| 0035_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.001 | 033_graphqa_nonlinear_0035_taskedge_count_seed44_magnetic_pooluniform_bias_lr0.001 | edge_count | mag-nonlinear-uniform+no-spd+rrwp+selfnode | uniform | 44 | 0.0010 | 0.3800 | 0.4200 | 0.3546 | 612.7595 | 1.0120 |
