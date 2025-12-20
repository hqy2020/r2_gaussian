# SPAGS 消融实验结果

## 实验配置
- **方法**: sps, adm, gar, sps_adm, sps_gar, gar_adm, spags
- **器官**: chest, foot, head, abdomen, pancreas  
- **视角**: 3, 6, 9
- **迭代**: 30000
- **指标**: 2D PSNR / SSIM
- **注**: spags 数据来自 optimized_spags_selection.json

---

## 3 views 结果

| 配置 | chest | foot | head | abdomen | pancreas | 平均 |
|------|-------|------|------|---------|----------|------|
| sps | 27.05 / 0.844 | 28.65 / 0.902 | 26.58 / 0.915 | 29.40 / 0.936 | 29.00 / 0.923 | 28.14 / 0.904 |
| adm | 26.42 / 0.840 | 28.69 / 0.898 | 26.78 / 0.923 | 29.31 / 0.937 | 29.00 / 0.922 | 28.04 / 0.904 |
| gar | 26.22 / 0.837 | 28.77 / 0.893 | 26.77 / 0.924 | 29.25 / 0.936 | 28.92 / 0.922 | 27.99 / 0.903 |
| sps_adm | 27.13 / 0.845 | 28.60 / 0.902 | 26.61 / 0.915 | 29.51 / 0.936 | 29.12 / 0.923 | 28.19 / 0.904 |
| sps_gar | 27.14 / 0.848 | 28.66 / 0.900 | 26.62 / 0.918 | 29.52 / 0.937 | 29.16 / 0.925 | 28.22 / 0.905 |
| gar_adm | 26.14 / 0.837 | 28.74 / 0.896 | 26.80 / 0.924 | 29.42 / 0.938 | 29.13 / 0.925 | 28.05 / 0.904 |
| spags | 27.33 / 0.848 | 28.83 / 0.900 | 26.78 / 0.918 | 29.66 / 0.938 | 29.13 / 0.924 | 28.35 / 0.906 |

## 6 views 结果

| 配置 | chest | foot | head | abdomen | pancreas | 平均 |
|------|-------|------|------|---------|----------|------|
| sps | 33.46 / 0.928 | 32.23 / 0.941 | 33.20 / 0.974 | 34.11 / 0.975 | 33.88 / 0.951 | 33.38 / 0.954 |
| adm | 33.42 / 0.927 | 32.61 / 0.938 | 33.03 / 0.973 | 34.22 / 0.974 | 33.60 / 0.951 | 33.38 / 0.953 |
| gar | 33.41 / 0.927 | 32.22 / 0.937 | 33.10 / 0.974 | 34.10 / 0.974 | 33.99 / 0.952 | 33.36 / 0.953 |
| sps_adm | 33.40 / 0.927 | 32.36 / 0.941 | 33.09 / 0.974 | 34.35 / 0.975 | 33.78 / 0.951 | 33.40 / 0.954 |
| sps_gar | 33.52 / 0.928 | 32.23 / 0.940 | 32.91 / 0.973 | 33.99 / 0.976 | 33.90 / 0.953 | 33.31 / 0.954 |
| gar_adm | 33.32 / 0.926 | 32.12 / 0.937 | 33.04 / 0.973 | 34.17 / 0.975 | 33.84 / 0.952 | 33.30 / 0.953 |
| spags | 33.47 / 0.928 | 32.38 / 0.940 | 32.88 / 0.973 | 34.37 / 0.976 | 33.91 / 0.952 | 33.40 / 0.954 |

## 9 views 结果

| 配置 | chest | foot | head | abdomen | pancreas | 平均 |
|------|-------|------|------|---------|----------|------|
| sps | 36.89 / 0.953 | 34.79 / 0.955 | 35.96 / 0.982 | 36.88 / 0.981 | 35.64 / 0.960 | 36.03 / 0.966 |
| adm | 37.05 / 0.954 | 35.06 / 0.954 | 35.88 / 0.982 | 36.97 / 0.982 | 35.55 / 0.960 | 36.10 / 0.966 |
| gar | 37.04 / 0.954 | 34.93 / 0.954 | 35.92 / 0.982 | 37.08 / 0.982 | 35.84 / 0.960 | 36.16 / 0.966 |
| sps_adm | 37.05 / 0.953 | 34.60 / 0.955 | 36.24 / 0.983 | 36.74 / 0.982 | 35.47 / 0.960 | 36.02 / 0.966 |
| sps_gar | 36.93 / 0.953 | 34.67 / 0.955 | 35.72 / 0.982 | 36.88 / 0.982 | 35.82 / 0.961 | 36.00 / 0.967 |
| gar_adm | 36.81 / 0.953 | 34.75 / 0.954 | 36.05 / 0.982 | 36.83 / 0.982 | 35.78 / 0.961 | 36.04 / 0.966 |
| spags | 36.79 / 0.952 | 34.74 / 0.955 | 36.20 / 0.982 | 36.74 / 0.982 | 35.70 / 0.961 | 36.03 / 0.967 |

## 平均值汇总

| 配置 | 3v PSNR | 3v SSIM | 6v PSNR | 6v SSIM | 9v PSNR | 9v SSIM |
|------|---------|---------|---------|---------|---------|---------|
| sps | 28.14 | 0.904 | 33.38 | 0.954 | 36.03 | 0.966 |
| adm | 28.04 | 0.904 | 33.38 | 0.953 | 36.10 | 0.966 |
| gar | 27.99 | 0.903 | 33.36 | 0.953 | 36.16 | 0.966 |
| sps_adm | 28.19 | 0.904 | 33.40 | 0.954 | 36.02 | 0.966 |
| sps_gar | 28.22 | 0.905 | 33.31 | 0.954 | 36.00 | 0.967 |
| gar_adm | 28.05 | 0.904 | 33.30 | 0.953 | 36.04 | 0.966 |
| spags | 28.35 | 0.906 | 33.40 | 0.954 | 36.03 | 0.967 |

---

## 完整实验路径 (105 个)

| # | 配置 | 器官 | 视角 | PSNR | SSIM | 输出路径 |
|---|------|------|------|------|------|----------|
| 1 | sps | chest | 3 | 27.05 | 0.844 | `output/_2025_12_18_01_01_chest_3views_sps` |
| 2 | sps | chest | 6 | 33.46 | 0.928 | `output/_2025_12_18_01_01_chest_6views_sps` |
| 3 | sps | chest | 9 | 36.89 | 0.953 | `output/_2025_12_18_01_01_chest_9views_sps` |
| 4 | sps | foot | 3 | 28.65 | 0.902 | `output/_2025_12_18_01_01_foot_3views_sps` |
| 5 | sps | foot | 6 | 32.23 | 0.941 | `output/sps_vs_spsadm_foot_6views_sps` |
| 6 | sps | foot | 9 | 34.79 | 0.955 | `output/_2025_12_18_01_01_foot_9views_sps` |
| 7 | sps | head | 3 | 26.58 | 0.915 | `output/_2025_12_18_01_01_head_3views_sps` |
| 8 | sps | head | 6 | 33.20 | 0.974 | `output/_2025_12_18_01_51_head_6views_sps` |
| 9 | sps | head | 9 | 35.96 | 0.982 | `output/_2025_12_18_01_51_head_9views_sps` |
| 10 | sps | abdomen | 3 | 29.40 | 0.936 | `output/_2025_12_18_01_53_abdomen_3views_sps` |
| 11 | sps | abdomen | 6 | 34.11 | 0.975 | `output/_2025_12_18_02_08_abdomen_6views_sps` |
| 12 | sps | abdomen | 9 | 36.88 | 0.981 | `output/_2025_12_18_02_08_abdomen_9views_sps` |
| 13 | sps | pancreas | 3 | 29.00 | 0.923 | `output/_2025_12_18_02_08_pancreas_3views_sps` |
| 14 | sps | pancreas | 6 | 33.88 | 0.951 | `output/_2025_12_18_02_13_pancreas_6views_sps` |
| 15 | sps | pancreas | 9 | 35.64 | 0.960 | `output/_2025_12_18_03_03_pancreas_9views_sps` |
| 16 | adm | chest | 3 | 26.42 | 0.840 | `output/_2025_12_12_23_44_chest_3views_adm` |
| 17 | adm | chest | 6 | 33.42 | 0.927 | `output/_2025_12_18_03_36_chest_6views_adm` |
| 18 | adm | chest | 9 | 37.05 | 0.954 | `output/_2025_12_12_23_44_chest_9views_adm` |
| 19 | adm | foot | 3 | 28.69 | 0.898 | `output/_2025_12_13_18_34_foot_3views_adm` |
| 20 | adm | foot | 6 | 32.61 | 0.938 | `output/_2025_12_15_00_27_foot_6views_adm` |
| 21 | adm | foot | 9 | 35.06 | 0.954 | `output/adm_fixed_1213_1305_foot_9views_adm` |
| 22 | adm | head | 3 | 26.78 | 0.923 | `output/_2025_12_12_23_44_head_3views_adm` |
| 23 | adm | head | 6 | 33.03 | 0.973 | `output/_2025_12_12_23_44_head_6views_adm` |
| 24 | adm | head | 9 | 35.88 | 0.982 | `output/_2025_12_12_23_44_head_9views_adm` |
| 25 | adm | abdomen | 3 | 29.31 | 0.937 | `output/_2025_12_12_23_44_abdomen_3views_adm` |
| 26 | adm | abdomen | 6 | 34.22 | 0.974 | `output/_2025_12_12_23_44_abdomen_6views_adm` |
| 27 | adm | abdomen | 9 | 36.97 | 0.982 | `output/_2025_12_12_23_44_abdomen_9views_adm` |
| 28 | adm | pancreas | 3 | 29.00 | 0.922 | `output/_2025_12_12_23_44_pancreas_3views_adm` |
| 29 | adm | pancreas | 6 | 33.60 | 0.951 | `output/_2025_12_13_18_36_pancreas_6views_adm` |
| 30 | adm | pancreas | 9 | 35.55 | 0.960 | `output/_2025_12_13_18_36_pancreas_9views_adm` |
| 31 | gar | chest | 3 | 26.22 | 0.837 | `output/gar_full_chest_3views_gar` |
| 32 | gar | chest | 6 | 33.41 | 0.927 | `output/_2025_12_18_07_55_chest_6views_gar` |
| 33 | gar | chest | 9 | 37.04 | 0.954 | `output/aa_2025_12_06_23_44_chest_9views_gar` |
| 34 | gar | foot | 3 | 28.77 | 0.893 | `output/2025_12_13_gar_rerun_foot_3views_gar` |
| 35 | gar | foot | 6 | 32.22 | 0.937 | `output/_2025_12_18_08_31_foot_6views_gar` |
| 36 | gar | foot | 9 | 34.93 | 0.954 | `output/gar_full_foot_9views_gar` |
| 37 | gar | head | 3 | 26.77 | 0.924 | `output/_2025_12_18_09_13_head_3views_gar` |
| 38 | gar | head | 6 | 33.10 | 0.974 | `output/aa_2025_12_07_14_59_head_6views_gar` |
| 39 | gar | head | 9 | 35.92 | 0.982 | `output/aa_2025_12_07_14_54_head_9views_gar` |
| 40 | gar | abdomen | 3 | 29.25 | 0.936 | `output/aa_2025_12_07_14_59_abdomen_3views_gar` |
| 41 | gar | abdomen | 6 | 34.10 | 0.974 | `output/_2025_12_20_12_32_abdomen_6views_gar` |
| 42 | gar | abdomen | 9 | 37.08 | 0.982 | `output/gar_full_abdomen_9views_gar` |
| 43 | gar | pancreas | 3 | 28.92 | 0.922 | `output/lucky_gar_pancreas_3views_gar` |
| 44 | gar | pancreas | 6 | 33.99 | 0.952 | `output/lucky_gar_pancreas_6views_gar` |
| 45 | gar | pancreas | 9 | 35.84 | 0.960 | `output/lucky_gar_pancreas_9views_gar` |
| 46 | sps_adm | chest | 3 | 27.13 | 0.845 | `output/_2025_12_18_13_44_chest_3views_sps_adm` |
| 47 | sps_adm | chest | 6 | 33.40 | 0.927 | `output/_2025_12_18_14_07_chest_6views_sps_adm` |
| 48 | sps_adm | chest | 9 | 37.05 | 0.953 | `output/_2025_12_18_13_20_chest_9views_sps_adm` |
| 49 | sps_adm | foot | 3 | 28.60 | 0.902 | `output/_2025_12_18_15_09_foot_3views_sps_adm` |
| 50 | sps_adm | foot | 6 | 32.36 | 0.941 | `output/sps_vs_spsadm_foot_6views_sps_adm` |
| 51 | sps_adm | foot | 9 | 34.60 | 0.955 | `output/_2025_12_19_10_29_foot_9views_sps_adm` |
| 52 | sps_adm | head | 3 | 26.61 | 0.915 | `output/_2025_12_18_15_28_head_3views_sps_adm` |
| 53 | sps_adm | head | 6 | 33.09 | 0.974 | `output/_2025_12_19_12_32_head_6views_sps_adm` |
| 54 | sps_adm | head | 9 | 36.24 | 0.983 | `output/_2025_12_19_14_36_head_9views_sps_adm` |
| 55 | sps_adm | abdomen | 3 | 29.51 | 0.936 | `output/_2025_12_18_16_43_abdomen_3views_sps_adm` |
| 56 | sps_adm | abdomen | 6 | 34.35 | 0.975 | `output/_2025_12_19_19_05_abdomen_6views_sps_adm` |
| 57 | sps_adm | abdomen | 9 | 36.74 | 0.982 | `output/_2025_12_19_22_19_abdomen_9views_sps_adm` |
| 58 | sps_adm | pancreas | 3 | 29.12 | 0.923 | `output/_2025_12_18_18_18_pancreas_3views_sps_adm` |
| 59 | sps_adm | pancreas | 6 | 33.78 | 0.951 | `output/_2025_12_20_03_39_pancreas_6views_sps_adm` |
| 60 | sps_adm | pancreas | 9 | 35.47 | 0.960 | `output/_2025_12_18_18_48_pancreas_9views_sps_adm` |
| 61 | sps_gar | chest | 3 | 27.14 | 0.848 | `output/_2025_12_17_14_27_chest_3views_sps_gar` |
| 62 | sps_gar | chest | 6 | 33.52 | 0.928 | `output/_2025_12_18_19_28_chest_6views_sps_gar` |
| 63 | sps_gar | chest | 9 | 36.93 | 0.953 | `output/_2025_12_19_10_37_chest_9views_sps_gar` |
| 64 | sps_gar | foot | 3 | 28.66 | 0.900 | `output/_2025_12_18_19_47_foot_3views_sps_gar` |
| 65 | sps_gar | foot | 6 | 32.23 | 0.940 | `output/_2025_12_19_12_18_foot_6views_sps_gar` |
| 66 | sps_gar | foot | 9 | 34.67 | 0.955 | `output/_2025_12_19_12_19_foot_9views_sps_gar` |
| 67 | sps_gar | head | 3 | 26.62 | 0.918 | `output/_2025_12_19_14_01_head_3views_sps_gar` |
| 68 | sps_gar | head | 6 | 32.91 | 0.973 | `output/_2025_12_19_15_22_head_6views_sps_gar` |
| 69 | sps_gar | head | 9 | 35.72 | 0.982 | `output/_2025_12_19_15_26_head_9views_sps_gar` |
| 70 | sps_gar | abdomen | 3 | 29.52 | 0.937 | `output/_2025_12_18_21_04_abdomen_3views_sps_gar` |
| 71 | sps_gar | abdomen | 6 | 33.99 | 0.976 | `output/_2025_12_19_16_04_abdomen_6views_sps_gar` |
| 72 | sps_gar | abdomen | 9 | 36.88 | 0.982 | `output/_2025_12_19_18_47_abdomen_9views_sps_gar` |
| 73 | sps_gar | pancreas | 3 | 29.16 | 0.925 | `output/_2025_12_20_02_41_pancreas_3views_sps_gar` |
| 74 | sps_gar | pancreas | 6 | 33.90 | 0.953 | `output/_2025_12_19_16_04_pancreas_6views_sps_gar` |
| 75 | sps_gar | pancreas | 9 | 35.82 | 0.961 | `output/_2025_12_20_06_55_pancreas_9views_sps_gar` |
| 76 | gar_adm | chest | 3 | 26.14 | 0.837 | `output/_2025_12_18_09_05_chest_3views_gar_adm` |
| 77 | gar_adm | chest | 6 | 33.32 | 0.926 | `output/_2025_12_19_10_37_chest_6views_gar_adm` |
| 78 | gar_adm | chest | 9 | 36.81 | 0.953 | `output/_2025_12_19_11_16_chest_9views_gar_adm` |
| 79 | gar_adm | foot | 3 | 28.74 | 0.896 | `output/_2025_12_19_11_55_foot_3views_gar_adm` |
| 80 | gar_adm | foot | 6 | 32.12 | 0.937 | `output/_2025_12_19_12_43_foot_6views_gar_adm` |
| 81 | gar_adm | foot | 9 | 34.75 | 0.954 | `output/_2025_12_19_13_49_foot_9views_gar_adm` |
| 82 | gar_adm | head | 3 | 26.80 | 0.924 | `output/_2025_12_17_14_27_head_3views_gar_adm` |
| 83 | gar_adm | head | 6 | 33.04 | 0.973 | `output/_2025_12_19_21_18_head_6views_gar_adm` |
| 84 | gar_adm | head | 9 | 36.05 | 0.982 | `output/_2025_12_20_01_38_head_9views_gar_adm` |
| 85 | gar_adm | abdomen | 3 | 29.42 | 0.938 | `output/_2025_12_20_04_59_abdomen_3views_gar_adm` |
| 86 | gar_adm | abdomen | 6 | 34.17 | 0.975 | `output/_2025_12_20_07_59_abdomen_6views_gar_adm` |
| 87 | gar_adm | abdomen | 9 | 36.83 | 0.982 | `output/_2025_12_19_23_00_abdomen_9views_gar_adm` |
| 88 | gar_adm | pancreas | 3 | 29.13 | 0.925 | `output/_2025_12_19_16_04_pancreas_3views_gar_adm` |
| 89 | gar_adm | pancreas | 6 | 33.84 | 0.952 | `output/_2025_12_20_01_47_pancreas_6views_gar_adm` |
| 90 | gar_adm | pancreas | 9 | 35.78 | 0.961 | `output/_2025_12_20_02_31_pancreas_9views_gar_adm` |
| 91 | spags | chest | 3 | 27.33 | 0.848 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_17_15_19_chest_3views_spags` |
| 92 | spags | chest | 6 | 33.47 | 0.928 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_10_37_chest_6views_spags` |
| 93 | spags | chest | 9 | 36.79 | 0.952 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_14_01_chest_9views_spags` |
| 94 | spags | foot | 3 | 28.83 | 0.900 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_12_04_14_51_foot_3views_spags` |
| 95 | spags | foot | 6 | 32.38 | 0.940 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_03_foot_6views_spags` |
| 96 | spags | foot | 9 | 34.74 | 0.955 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_02_11_foot_9views_spags` |
| 97 | spags | head | 3 | 26.78 | 0.918 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_17_15_19_head_3views_spags` |
| 98 | spags | head | 6 | 32.88 | 0.973 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_20_22_head_6views_spags` |
| 99 | spags | head | 9 | 36.20 | 0.982 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_52_head_9views_spags` |
| 100 | spags | abdomen | 3 | 29.66 | 0.938 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_abdomen_3views_spags` |
| 101 | spags | abdomen | 6 | 34.37 | 0.976 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_17_abdomen_6views_spags` |
| 102 | spags | abdomen | 9 | 36.74 | 0.982 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_09_30_abdomen_9views_spags` |
| 103 | spags | pancreas | 3 | 29.13 | 0.924 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_pancreas_3views_spags` |
| 104 | spags | pancreas | 6 | 33.91 | 0.952 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_20_29_pancreas_6views_spags` |
| 105 | spags | pancreas | 9 | 35.70 | 0.961 | `/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_23_03_pancreas_9views_spags` |
