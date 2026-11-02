# Agent-loop prompt cache: google/gemma-4-E4B-it (bf16, thinking on)

`scripts/agent_loop_bench.py` @ `69e45fa`, MLX 0.32.2, Apple M4 Max. 16 turns, system prompt 2500 tokens, tool output 700 tokens per turn, 24 greedy tokens per reply; cache layers {'RotatingKVCache': 20, 'KVCache': 4}. server: mlx_lm 0.31.3 server policy (LRU of 10 cache copies at the system boundary, the thinking tail and after each reply; prefix reuse only); server+msg: the same with a copy at every message boundary. rewind: 2 trunk(s), exact trims for full-attention layers, checkpoints of the other layers at the system boundary, at the last 3 message boundaries and every 4096 tokens, replay from the nearest checkpoint. Prefill seconds include the cache copies the server makes.

| scenario | policy | final prompt | tokens processed, all turns | prefill s, all turns | median turn s | worst turn s | store GB at end | peak GB | replies identical |
|---|---|---|---|---|---|---|---|---|---|
| append | server | 14375 | 71852 | 29.9 | 1.55 | 4.14 | 1.97 | 17.7 | yes |
| append | rewind | 14375 | 14364 | 7.1 | 0.39 | 1.34 | 0.51 | 16.1 | no |
| edit | server | 9277 | 62223 | 26.8 | 1.61 | 2.91 | 1.42 | 17.4 | yes |
| edit | rewind | 9277 | 20065 | 10.0 | 0.58 | 1.34 | 0.52 | 16.0 | no |
| compact | server | 6265 | 64776 | 28.5 | 1.65 | 2.92 | 1.35 | 17.4 | yes |
| compact | rewind | 6265 | 18895 | 9.7 | 0.41 | 1.63 | 0.36 | 15.9 | no |

Per turn (prefill seconds; tokens processed in brackets):

**append**

| turn | prompt tokens | server | rewind |
|---|---|---|---|
| 0 | 3250 | 1.20 (3249) | 1.34 (3249) |
| 1 | 3991 | 0.54 (1460) | 0.34 (743) |
| 2 | 4732 | 0.83 (2201) | 0.37 (743) |
| 3 | 5473 | 1.09 (2942) | 0.36 (743) |
| 4 | 6214 | 1.37 (3683) | 0.38 (743) |
| 5 | 6955 | 1.74 (4424) | 0.39 (743) |
| 6 | 7695 | 2.15 (5164) | 0.39 (742) |
| 7 | 8436 | 2.54 (5905) | 0.41 (743) |
| 8 | 9177 | 2.86 (6646) | 0.39 (743) |
| 9 | 9918 | 3.16 (7387) | 0.39 (743) |
| 10 | 10660 | 3.53 (8129) | 0.39 (744) |
| 11 | 11403 | 3.83 (8872) | 0.40 (745) |
| 12 | 12146 | 4.14 (9615) | 0.40 (745) |
| 13 | 12889 | 0.32 (725) | 0.41 (745) |
| 14 | 13632 | 0.32 (725) | 0.36 (725) |
| 15 | 14375 | 0.33 (725) | 0.36 (725) |

**edit**

| turn | prompt tokens | server | rewind |
|---|---|---|---|
| 0 | 3250 | 1.43 (3249) | 1.34 (3249) |
| 1 | 3641 | 0.44 (1110) | 0.51 (1113) |
| 2 | 4031 | 0.61 (1500) | 0.54 (1112) |
| 3 | 4434 | 0.84 (1903) | 0.61 (1125) |
| 4 | 4839 | 1.05 (2308) | 0.58 (1127) |
| 5 | 5214 | 1.16 (2683) | 0.57 (1097) |
| 6 | 5646 | 1.35 (3115) | 0.61 (1154) |
| 7 | 6050 | 1.51 (3519) | 0.58 (1125) |
| 8 | 6449 | 1.70 (3918) | 0.57 (1121) |
| 9 | 6882 | 1.87 (4351) | 0.60 (1155) |
| 10 | 7287 | 2.05 (4756) | 0.59 (1127) |
| 11 | 7721 | 2.23 (5190) | 0.61 (1157) |
| 12 | 8125 | 2.39 (5594) | 0.57 (1127) |
| 13 | 8472 | 2.55 (5941) | 0.59 (1070) |
| 14 | 8871 | 2.74 (6340) | 0.59 (1122) |
| 15 | 9277 | 2.91 (6746) | 0.57 (1084) |

**compact**

| turn | prompt tokens | server | rewind |
|---|---|---|---|
| 0 | 3250 | 1.50 (3249) | 1.38 (3249) |
| 1 | 3991 | 0.59 (1460) | 0.35 (743) |
| 2 | 4732 | 1.02 (2201) | 0.37 (743) |
| 3 | 5473 | 1.32 (2942) | 0.39 (743) |
| 4 | 6214 | 1.63 (3683) | 0.41 (743) |
| 5 | 6955 | 1.98 (4424) | 0.41 (743) |
| 6 | 7695 | 2.24 (5164) | 0.41 (742) |
| 7 | 8436 | 2.59 (5905) | 0.42 (743) |
| 8 | 5513 | 1.26 (2982) | 1.61 (2985) |
| 9 | 6254 | 1.65 (3723) | 0.36 (743) |
| 10 | 6996 | 1.99 (4465) | 0.37 (744) |
| 11 | 7739 | 2.27 (5208) | 0.39 (745) |
| 12 | 8482 | 2.58 (5951) | 0.44 (745) |
| 13 | 9225 | 2.92 (6694) | 0.41 (745) |
| 14 | 5522 | 1.30 (2991) | 1.63 (2994) |
| 15 | 6265 | 1.64 (3734) | 0.36 (745) |

