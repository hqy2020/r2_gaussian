# 6 种方法 × 15 场景 2D 指标对比表
# (PSNR/SSIM @ 30k iterations)
# 生成时间: 2025-12-20


## 3 Views

| 方法 | chest | foot | head | abdomen | pancreas | **平均** |
|------|-------|------|------|---------|----------|----------|
| SPAGS | 27.03/0.847 | 28.59/0.900 | 26.75/0.919 | 29.66/0.938 | 29.13/0.924 | **28.23/0.905** |
| baseline | 26.16/0.837 | 28.82/0.898 | 26.59/0.922 | 29.24/0.936 | 28.58/0.920 | **27.88/0.903** |
| xgaussian | 20.48/0.739 | 26.03/0.848 | 21.02/0.863 | 23.56/0.906 | 24.93/0.898 | **23.20/0.851** |
| fsgs | 20.55/0.736 | 25.79/0.851 | 20.51/0.859 | 24.01/0.907 | 25.08/0.899 | **23.19/0.851** |
| dngaussian | 20.52/0.748 | 24.78/0.852 | 17.70/0.798 | 16.20/0.827 | 23.65/0.885 | **20.57/0.822** |
| corgs | 19.53/0.724 | 25.25/0.851 | 20.54/0.862 | 22.59/0.909 | 20.36/0.880 | **21.65/0.845** |

## 6 Views

| 方法 | chest | foot | head | abdomen | pancreas | **平均** |
|------|-------|------|------|---------|----------|----------|
| SPAGS | 33.44/0.928 | 32.31/0.940 | 32.78/0.973 | 34.37/0.976 | 33.91/0.952 | **33.36/0.954** |
| baseline | 33.14/0.927 | 32.31/0.937 | 33.03/0.973 | 34.00/0.974 | 33.40/0.950 | **33.18/0.952** |
| xgaussian | 28.37/0.877 | 28.28/0.895 | 27.48/0.942 | 27.41/0.946 | 29.03/0.932 | **28.11/0.918** |
| fsgs | 27.61/0.876 | 28.34/0.897 | 27.77/0.944 | 28.17/0.950 | 28.82/0.933 | **28.14/0.920** |
| dngaussian | 27.34/0.863 | 27.57/0.901 | 18.24/0.836 | 19.73/0.907 | 28.97/0.933 | **24.37/0.888** |
| corgs | 27.69/0.871 | 28.45/0.901 | 27.28/0.944 | 28.27/0.951 | 29.28/0.936 | **28.19/0.921** |

## 9 Views

| 方法 | chest | foot | head | abdomen | pancreas | **平均** |
|------|-------|------|------|---------|----------|----------|
| SPAGS | 36.79/0.952 | 34.48/0.955 | 36.15/0.982 | 36.74/0.982 | 35.70/0.961 | **35.97/0.967** |
| baseline | 36.92/0.953 | 34.96/0.955 | 35.80/0.982 | 36.97/0.981 | 35.80/0.960 | **36.09/0.966** |
| xgaussian | 31.90/0.915 | 29.78/0.922 | 29.41/0.956 | 30.11/0.960 | 29.81/0.944 | **30.20/0.939** |
| fsgs | 31.29/0.915 | 30.74/0.926 | 30.22/0.960 | 31.46/0.964 | 31.03/0.945 | **30.95/0.942** |
| dngaussian | 30.42/0.898 | 29.55/0.921 | 21.16/0.872 | 21.40/0.924 | 29.77/0.943 | **26.46/0.912** |
| corgs | 32.26/0.921 | 30.89/0.928 | 30.02/0.956 | 31.46/0.965 | 30.78/0.947 | **31.08/0.943** |

---

## 90 条实验完整路径

| # | 方法 | 器官 | 视角 | PSNR | SSIM | Output Path |
|---|------|------|------|------|------|-------------|
| 1 | spags | chest | 3v | 27.03 | 0.847 | `_2025_12_18_09_04_chest_3views_spags` |
| 2 | spags | foot | 3v | 28.59 | 0.900 | `_2025_12_19_16_08_foot_3views_spags` |
| 3 | spags | head | 3v | 26.75 | 0.919 | `_2025_12_20_05_42_head_3views_spags` |
| 4 | spags | abdomen | 3v | 29.66 | 0.938 | `_2025_12_19_16_04_abdomen_3views_spags` |
| 5 | spags | pancreas | 3v | 29.13 | 0.924 | `_2025_12_19_16_04_pancreas_3views_spags` |
| 6 | spags | chest | 6v | 33.44 | 0.928 | `_2025_12_19_10_37_chest_6views_spags` |
| 7 | spags | foot | 6v | 32.31 | 0.940 | `_2025_12_19_22_06_foot_6views_spags` |
| 8 | spags | head | 6v | 32.78 | 0.973 | `_2025_12_20_08_14_head_6views_spags` |
| 9 | spags | abdomen | 6v | 34.37 | 0.976 | `_2025_12_19_22_17_abdomen_6views_spags` |
| 10 | spags | pancreas | 6v | 33.91 | 0.952 | `_2025_12_19_20_29_pancreas_6views_spags` |
| 11 | spags | chest | 9v | 36.79 | 0.952 | `_2025_12_19_14_01_chest_9views_spags` |
| 12 | spags | foot | 9v | 34.48 | 0.955 | `_2025_12_20_02_16_foot_9views_spags` |
| 13 | spags | head | 9v | 36.15 | 0.982 | `_2025_12_19_22_52_head_9views_spags` |
| 14 | spags | abdomen | 9v | 36.74 | 0.982 | `_2025_12_20_09_30_abdomen_9views_spags` |
| 15 | spags | pancreas | 9v | 35.70 | 0.961 | `_2025_12_19_23_03_pancreas_9views_spags` |
| 16 | baseline | chest | 3v | 26.16 | 0.837 | `_2025_12_19_11_00_chest_3views_baseline` |
| 17 | baseline | foot | 3v | 28.82 | 0.898 | `_2025_12_18_12_19_foot_3views_baseline` |
| 18 | baseline | head | 3v | 26.59 | 0.922 | `_2025_12_18_14_46_head_3views_baseline` |
| 19 | baseline | abdomen | 3v | 29.24 | 0.936 | `_2025_12_18_19_28_abdomen_3views_baseline` |
| 20 | baseline | pancreas | 3v | 28.58 | 0.920 | `_2025_12_18_14_32_pancreas_3views_baseline` |
| 21 | baseline | chest | 6v | 33.14 | 0.927 | `_2025_12_18_11_57_chest_6views_baseline` |
| 22 | baseline | foot | 6v | 32.31 | 0.937 | `_2025_12_18_12_23_foot_6views_baseline` |
| 23 | baseline | head | 6v | 33.03 | 0.973 | `_2025_12_18_16_32_head_6views_baseline` |
| 24 | baseline | abdomen | 6v | 34.00 | 0.974 | `_2025_12_18_13_40_abdomen_6views_baseline` |
| 25 | baseline | pancreas | 6v | 33.40 | 0.950 | `_2025_12_18_14_34_pancreas_6views_baseline` |
| 26 | baseline | chest | 9v | 36.92 | 0.953 | `_2025_12_18_12_05_chest_9views_baseline` |
| 27 | baseline | foot | 9v | 34.96 | 0.955 | `_2025_12_18_13_27_foot_9views_baseline` |
| 28 | baseline | head | 9v | 35.80 | 0.982 | `_2025_12_18_18_02_head_9views_baseline` |
| 29 | baseline | abdomen | 9v | 36.97 | 0.981 | `_2025_12_18_14_00_abdomen_9views_baseline` |
| 30 | baseline | pancreas | 9v | 35.80 | 0.960 | `_2025_12_18_14_50_pancreas_9views_baseline` |
| 31 | xgaussian | chest | 3v | 20.48 | 0.739 | `_2025_12_19_11_57_chest_3views_xgaussian` |
| 32 | xgaussian | foot | 3v | 26.03 | 0.848 | `_2025_12_18_10_03_foot_3views_xgaussian` |
| 33 | xgaussian | head | 3v | 21.02 | 0.863 | `_2025_12_18_11_21_head_3views_xgaussian` |
| 34 | xgaussian | abdomen | 3v | 23.56 | 0.906 | `_2025_12_18_13_43_abdomen_3views_xgaussian` |
| 35 | xgaussian | pancreas | 3v | 24.93 | 0.898 | `_2025_12_18_18_00_pancreas_3views_xgaussian` |
| 36 | xgaussian | chest | 6v | 28.37 | 0.877 | `_2025_12_18_09_35_chest_6views_xgaussian` |
| 37 | xgaussian | foot | 6v | 28.28 | 0.895 | `_2025_12_18_10_24_foot_6views_xgaussian` |
| 38 | xgaussian | head | 6v | 27.48 | 0.942 | `_2025_12_18_12_02_head_6views_xgaussian` |
| 39 | xgaussian | abdomen | 6v | 27.41 | 0.946 | `_2025_12_18_15_06_abdomen_6views_xgaussian` |
| 40 | xgaussian | pancreas | 6v | 29.03 | 0.932 | `_2025_12_18_19_18_pancreas_6views_xgaussian` |
| 41 | xgaussian | chest | 9v | 31.90 | 0.915 | `_2025_12_18_09_48_chest_9views_xgaussian` |
| 42 | xgaussian | foot | 9v | 29.78 | 0.922 | `_2025_12_18_10_51_foot_9views_xgaussian` |
| 43 | xgaussian | head | 9v | 29.41 | 0.956 | `_2025_12_18_12_44_head_9views_xgaussian` |
| 44 | xgaussian | abdomen | 9v | 30.11 | 0.960 | `_2025_12_18_16_41_abdomen_9views_xgaussian` |
| 45 | xgaussian | pancreas | 9v | 29.81 | 0.944 | `_2025_12_18_20_31_pancreas_9views_xgaussian` |
| 46 | fsgs | chest | 3v | 20.55 | 0.736 | `_2025_12_19_12_26_chest_3views_fsgs` |
| 47 | fsgs | foot | 3v | 25.79 | 0.851 | `_2025_12_18_00_40_foot_3views_fsgs` |
| 48 | fsgs | head | 3v | 20.51 | 0.859 | `_2025_12_18_00_51_head_3views_fsgs` |
| 49 | fsgs | abdomen | 3v | 24.01 | 0.907 | `_2025_12_18_00_55_abdomen_3views_fsgs` |
| 50 | fsgs | pancreas | 3v | 25.08 | 0.899 | `_2025_12_18_01_18_pancreas_3views_fsgs` |
| 51 | fsgs | chest | 6v | 27.61 | 0.876 | `_2025_12_18_00_40_chest_6views_fsgs` |
| 52 | fsgs | foot | 6v | 28.34 | 0.897 | `_2025_12_18_00_40_foot_6views_fsgs` |
| 53 | fsgs | head | 6v | 27.77 | 0.944 | `_2025_12_18_00_54_head_6views_fsgs` |
| 54 | fsgs | abdomen | 6v | 28.17 | 0.950 | `_2025_12_18_01_00_abdomen_6views_fsgs` |
| 55 | fsgs | pancreas | 6v | 28.82 | 0.933 | `_2025_12_18_01_21_pancreas_6views_fsgs` |
| 56 | fsgs | chest | 9v | 31.29 | 0.915 | `_2025_12_18_00_40_chest_9views_fsgs` |
| 57 | fsgs | foot | 9v | 30.74 | 0.926 | `_2025_12_18_00_50_foot_9views_fsgs` |
| 58 | fsgs | head | 9v | 30.22 | 0.960 | `_2025_12_18_00_55_head_9views_fsgs` |
| 59 | fsgs | abdomen | 9v | 31.46 | 0.964 | `_2025_12_18_01_16_abdomen_9views_fsgs` |
| 60 | fsgs | pancreas | 9v | 31.03 | 0.945 | `_2025_12_18_01_36_pancreas_9views_fsgs` |
| 61 | dngaussian | chest | 3v | 20.52 | 0.748 | `_2025_12_20_21_02_chest_3views_dngaussian` |
| 62 | dngaussian | foot | 3v | 24.78 | 0.852 | `_2025_12_20_21_02_foot_3views_dngaussian` |
| 63 | dngaussian | head | 3v | 17.70 | 0.798 | `_2025_12_20_21_02_head_3views_dngaussian` |
| 64 | dngaussian | abdomen | 3v | 16.20 | 0.827 | `_2025_12_20_21_02_abdomen_3views_dngaussian` |
| 65 | dngaussian | pancreas | 3v | 23.65 | 0.885 | `_2025_12_20_21_02_pancreas_3views_dngaussian` |
| 66 | dngaussian | chest | 6v | 27.34 | 0.863 | `_2025_12_20_21_02_chest_6views_dngaussian` |
| 67 | dngaussian | foot | 6v | 27.57 | 0.901 | `_2025_12_20_21_02_foot_6views_dngaussian` |
| 68 | dngaussian | head | 6v | 18.24 | 0.836 | `_2025_12_20_21_02_head_6views_dngaussian` |
| 69 | dngaussian | abdomen | 6v | 19.73 | 0.907 | `_2025_12_20_21_02_abdomen_6views_dngaussian` |
| 70 | dngaussian | pancreas | 6v | 28.97 | 0.933 | `_2025_12_20_21_02_pancreas_6views_dngaussian` |
| 71 | dngaussian | chest | 9v | 30.42 | 0.898 | `_2025_12_20_21_02_chest_9views_dngaussian` |
| 72 | dngaussian | foot | 9v | 29.55 | 0.921 | `_2025_12_20_21_02_foot_9views_dngaussian` |
| 73 | dngaussian | head | 9v | 21.16 | 0.872 | `_2025_12_19_16_04_head_9views_dngaussian` |
| 74 | dngaussian | abdomen | 9v | 21.40 | 0.924 | `_2025_12_20_21_02_abdomen_9views_dngaussian` |
| 75 | dngaussian | pancreas | 9v | 29.77 | 0.943 | `_2025_12_19_16_04_pancreas_9views_dngaussian` |
| 76 | corgs | chest | 3v | 19.53 | 0.724 | `_2025_12_20_13_10_chest_3views_corgs` |
| 77 | corgs | foot | 3v | 25.25 | 0.851 | `_2025_12_20_13_10_foot_3views_corgs` |
| 78 | corgs | head | 3v | 20.54 | 0.862 | `_2025_12_20_13_10_head_3views_corgs` |
| 79 | corgs | abdomen | 3v | 22.59 | 0.909 | `_2025_12_20_13_10_abdomen_3views_corgs` |
| 80 | corgs | pancreas | 3v | 20.36 | 0.880 | `_2025_12_20_13_10_pancreas_3views_corgs` |
| 81 | corgs | chest | 6v | 27.69 | 0.871 | `_2025_12_20_13_10_chest_6views_corgs` |
| 82 | corgs | foot | 6v | 28.45 | 0.901 | `_2025_12_20_13_10_foot_6views_corgs` |
| 83 | corgs | head | 6v | 27.28 | 0.944 | `_2025_12_20_13_10_head_6views_corgs` |
| 84 | corgs | abdomen | 6v | 28.27 | 0.951 | `_2025_12_19_16_04_abdomen_6views_corgs` |
| 85 | corgs | pancreas | 6v | 29.28 | 0.936 | `_2025_12_20_13_10_pancreas_6views_corgs` |
| 86 | corgs | chest | 9v | 32.26 | 0.921 | `_2025_12_20_13_10_chest_9views_corgs` |
| 87 | corgs | foot | 9v | 30.89 | 0.928 | `_2025_12_18_08_14_foot_9views_corgs` |
| 88 | corgs | head | 9v | 30.02 | 0.956 | `_2025_12_20_13_10_head_9views_corgs` |
| 89 | corgs | abdomen | 9v | 31.46 | 0.965 | `_2025_12_18_08_50_abdomen_9views_corgs` |
| 90 | corgs | pancreas | 9v | 30.78 | 0.947 | `_2025_12_20_13_10_pancreas_9views_corgs` |

---

## JSON 格式数据

```json
[
  {
    "method": "spags",
    "organ": "chest",
    "views": 3,
    "psnr": 27.033132553100586,
    "ssim": 0.8473542928695679,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_09_04_chest_3views_spags"
  },
  {
    "method": "spags",
    "organ": "foot",
    "views": 3,
    "psnr": 28.5860538482666,
    "ssim": 0.8995616436004639,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_08_foot_3views_spags"
  },
  {
    "method": "spags",
    "organ": "head",
    "views": 3,
    "psnr": 26.753684997558594,
    "ssim": 0.9185286164283752,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_05_42_head_3views_spags"
  },
  {
    "method": "spags",
    "organ": "abdomen",
    "views": 3,
    "psnr": 29.65963363647461,
    "ssim": 0.9376375675201416,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_abdomen_3views_spags"
  },
  {
    "method": "spags",
    "organ": "pancreas",
    "views": 3,
    "psnr": 29.1285343170166,
    "ssim": 0.9243782162666321,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_pancreas_3views_spags"
  },
  {
    "method": "spags",
    "organ": "chest",
    "views": 6,
    "psnr": 33.44194412231445,
    "ssim": 0.9277744889259338,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_10_37_chest_6views_spags"
  },
  {
    "method": "spags",
    "organ": "foot",
    "views": 6,
    "psnr": 32.310367584228516,
    "ssim": 0.9401980638504028,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_06_foot_6views_spags"
  },
  {
    "method": "spags",
    "organ": "head",
    "views": 6,
    "psnr": 32.78178787231445,
    "ssim": 0.9734495282173157,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_08_14_head_6views_spags"
  },
  {
    "method": "spags",
    "organ": "abdomen",
    "views": 6,
    "psnr": 34.37041091918945,
    "ssim": 0.9755123257637024,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_17_abdomen_6views_spags"
  },
  {
    "method": "spags",
    "organ": "pancreas",
    "views": 6,
    "psnr": 33.91228103637695,
    "ssim": 0.9524381756782532,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_20_29_pancreas_6views_spags"
  },
  {
    "method": "spags",
    "organ": "chest",
    "views": 9,
    "psnr": 36.794898986816406,
    "ssim": 0.9524509310722351,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_14_01_chest_9views_spags"
  },
  {
    "method": "spags",
    "organ": "foot",
    "views": 9,
    "psnr": 34.48004913330078,
    "ssim": 0.9548546671867371,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_02_16_foot_9views_spags"
  },
  {
    "method": "spags",
    "organ": "head",
    "views": 9,
    "psnr": 36.15178298950195,
    "ssim": 0.9824314117431641,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_52_head_9views_spags"
  },
  {
    "method": "spags",
    "organ": "abdomen",
    "views": 9,
    "psnr": 36.7425651550293,
    "ssim": 0.9820456504821777,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_09_30_abdomen_9views_spags"
  },
  {
    "method": "spags",
    "organ": "pancreas",
    "views": 9,
    "psnr": 35.696563720703125,
    "ssim": 0.9610289931297302,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_23_03_pancreas_9views_spags"
  },
  {
    "method": "baseline",
    "organ": "chest",
    "views": 3,
    "psnr": 26.15926742553711,
    "ssim": 0.836825966835022,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_11_00_chest_3views_baseline"
  },
  {
    "method": "baseline",
    "organ": "foot",
    "views": 3,
    "psnr": 28.81786346435547,
    "ssim": 0.8981672525405884,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_19_foot_3views_baseline"
  },
  {
    "method": "baseline",
    "organ": "head",
    "views": 3,
    "psnr": 26.589521408081055,
    "ssim": 0.9221019148826599,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_46_head_3views_baseline"
  },
  {
    "method": "baseline",
    "organ": "abdomen",
    "views": 3,
    "psnr": 29.23626136779785,
    "ssim": 0.9361777305603027,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_19_28_abdomen_3views_baseline"
  },
  {
    "method": "baseline",
    "organ": "pancreas",
    "views": 3,
    "psnr": 28.5755615234375,
    "ssim": 0.9197074174880981,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_32_pancreas_3views_baseline"
  },
  {
    "method": "baseline",
    "organ": "chest",
    "views": 6,
    "psnr": 33.14496612548828,
    "ssim": 0.9266059398651123,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_11_57_chest_6views_baseline"
  },
  {
    "method": "baseline",
    "organ": "foot",
    "views": 6,
    "psnr": 32.30937194824219,
    "ssim": 0.9374606609344482,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_23_foot_6views_baseline"
  },
  {
    "method": "baseline",
    "organ": "head",
    "views": 6,
    "psnr": 33.028873443603516,
    "ssim": 0.9734086394309998,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_16_32_head_6views_baseline"
  },
  {
    "method": "baseline",
    "organ": "abdomen",
    "views": 6,
    "psnr": 34.00027084350586,
    "ssim": 0.9741194844245911,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_13_40_abdomen_6views_baseline"
  },
  {
    "method": "baseline",
    "organ": "pancreas",
    "views": 6,
    "psnr": 33.403053283691406,
    "ssim": 0.9503161907196045,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_34_pancreas_6views_baseline"
  },
  {
    "method": "baseline",
    "organ": "chest",
    "views": 9,
    "psnr": 36.919403076171875,
    "ssim": 0.9534459114074707,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_05_chest_9views_baseline"
  },
  {
    "method": "baseline",
    "organ": "foot",
    "views": 9,
    "psnr": 34.95683288574219,
    "ssim": 0.9545350670814514,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_13_27_foot_9views_baseline"
  },
  {
    "method": "baseline",
    "organ": "head",
    "views": 9,
    "psnr": 35.800086975097656,
    "ssim": 0.9820998311042786,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_18_02_head_9views_baseline"
  },
  {
    "method": "baseline",
    "organ": "abdomen",
    "views": 9,
    "psnr": 36.9666748046875,
    "ssim": 0.98125159740448,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_00_abdomen_9views_baseline"
  },
  {
    "method": "baseline",
    "organ": "pancreas",
    "views": 9,
    "psnr": 35.80122375488281,
    "ssim": 0.960427463054657,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_50_pancreas_9views_baseline"
  },
  {
    "method": "xgaussian",
    "organ": "chest",
    "views": 3,
    "psnr": 20.47929573059082,
    "ssim": 0.7394033074378967,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_11_57_chest_3views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "foot",
    "views": 3,
    "psnr": 26.02786636352539,
    "ssim": 0.848497748374939,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_10_03_foot_3views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "head",
    "views": 3,
    "psnr": 21.01629638671875,
    "ssim": 0.8627992272377014,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_11_21_head_3views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "abdomen",
    "views": 3,
    "psnr": 23.555665969848633,
    "ssim": 0.9057514667510986,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_13_43_abdomen_3views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "pancreas",
    "views": 3,
    "psnr": 24.933164596557617,
    "ssim": 0.8977657556533813,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_18_00_pancreas_3views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "chest",
    "views": 6,
    "psnr": 28.368423461914062,
    "ssim": 0.876925528049469,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_09_35_chest_6views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "foot",
    "views": 6,
    "psnr": 28.276782989501953,
    "ssim": 0.8945271372795105,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_10_24_foot_6views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "head",
    "views": 6,
    "psnr": 27.483203887939453,
    "ssim": 0.9423269629478455,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_02_head_6views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "abdomen",
    "views": 6,
    "psnr": 27.407129287719727,
    "ssim": 0.9461808204650879,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_15_06_abdomen_6views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "pancreas",
    "views": 6,
    "psnr": 29.026451110839844,
    "ssim": 0.9319986701011658,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_19_18_pancreas_6views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "chest",
    "views": 9,
    "psnr": 31.895437240600586,
    "ssim": 0.9152273535728455,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_09_48_chest_9views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "foot",
    "views": 9,
    "psnr": 29.783527374267578,
    "ssim": 0.921653687953949,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_10_51_foot_9views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "head",
    "views": 9,
    "psnr": 29.406208038330078,
    "ssim": 0.9556609988212585,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_44_head_9views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "abdomen",
    "views": 9,
    "psnr": 30.112205505371094,
    "ssim": 0.9604483246803284,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_16_41_abdomen_9views_xgaussian"
  },
  {
    "method": "xgaussian",
    "organ": "pancreas",
    "views": 9,
    "psnr": 29.809226989746094,
    "ssim": 0.9436275362968445,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_20_31_pancreas_9views_xgaussian"
  },
  {
    "method": "fsgs",
    "organ": "chest",
    "views": 3,
    "psnr": 20.549013137817383,
    "ssim": 0.7364667654037476,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_12_26_chest_3views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "foot",
    "views": 3,
    "psnr": 25.794198989868164,
    "ssim": 0.8513047695159912,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_40_foot_3views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "head",
    "views": 3,
    "psnr": 20.507638931274414,
    "ssim": 0.858717679977417,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_51_head_3views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "abdomen",
    "views": 3,
    "psnr": 24.011011123657227,
    "ssim": 0.9072802662849426,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_55_abdomen_3views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "pancreas",
    "views": 3,
    "psnr": 25.078950881958008,
    "ssim": 0.8987756967544556,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_18_pancreas_3views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "chest",
    "views": 6,
    "psnr": 27.609996795654297,
    "ssim": 0.8757072687149048,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_40_chest_6views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "foot",
    "views": 6,
    "psnr": 28.341354370117188,
    "ssim": 0.8967141509056091,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_40_foot_6views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "head",
    "views": 6,
    "psnr": 27.77044677734375,
    "ssim": 0.9437386393547058,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_54_head_6views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "abdomen",
    "views": 6,
    "psnr": 28.1696834564209,
    "ssim": 0.9500167369842529,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_00_abdomen_6views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "pancreas",
    "views": 6,
    "psnr": 28.822328567504883,
    "ssim": 0.9331287145614624,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_21_pancreas_6views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "chest",
    "views": 9,
    "psnr": 31.285165786743164,
    "ssim": 0.914865255355835,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_40_chest_9views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "foot",
    "views": 9,
    "psnr": 30.743650436401367,
    "ssim": 0.92561936378479,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_50_foot_9views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "head",
    "views": 9,
    "psnr": 30.221412658691406,
    "ssim": 0.9595415592193604,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_55_head_9views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "abdomen",
    "views": 9,
    "psnr": 31.458993911743164,
    "ssim": 0.963881254196167,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_16_abdomen_9views_fsgs"
  },
  {
    "method": "fsgs",
    "organ": "pancreas",
    "views": 9,
    "psnr": 31.0325870513916,
    "ssim": 0.9453928470611572,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_36_pancreas_9views_fsgs"
  },
  {
    "method": "dngaussian",
    "organ": "chest",
    "views": 3,
    "psnr": 20.521177291870117,
    "ssim": 0.748060941696167,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_chest_3views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "foot",
    "views": 3,
    "psnr": 24.783300399780273,
    "ssim": 0.8519673347473145,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_foot_3views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "head",
    "views": 3,
    "psnr": 17.69994354248047,
    "ssim": 0.7979041934013367,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_head_3views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "abdomen",
    "views": 3,
    "psnr": 16.202392578125,
    "ssim": 0.8273808360099792,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_abdomen_3views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "pancreas",
    "views": 3,
    "psnr": 23.651145935058594,
    "ssim": 0.885021448135376,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_pancreas_3views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "chest",
    "views": 6,
    "psnr": 27.336259841918945,
    "ssim": 0.8632784485816956,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_chest_6views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "foot",
    "views": 6,
    "psnr": 27.567790985107422,
    "ssim": 0.9006609320640564,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_foot_6views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "head",
    "views": 6,
    "psnr": 18.238021850585938,
    "ssim": 0.8356928825378418,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_head_6views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "abdomen",
    "views": 6,
    "psnr": 19.72917366027832,
    "ssim": 0.9066805839538574,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_abdomen_6views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "pancreas",
    "views": 6,
    "psnr": 28.968687057495117,
    "ssim": 0.9326748847961426,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_pancreas_6views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "chest",
    "views": 9,
    "psnr": 30.420761108398438,
    "ssim": 0.8976189494132996,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_chest_9views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "foot",
    "views": 9,
    "psnr": 29.551424026489258,
    "ssim": 0.9214521050453186,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_foot_9views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "head",
    "views": 9,
    "psnr": 21.156824111938477,
    "ssim": 0.8720898628234863,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_head_9views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "abdomen",
    "views": 9,
    "psnr": 21.400060653686523,
    "ssim": 0.9237166047096252,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_abdomen_9views_dngaussian"
  },
  {
    "method": "dngaussian",
    "organ": "pancreas",
    "views": 9,
    "psnr": 29.774702072143555,
    "ssim": 0.9432757496833801,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_pancreas_9views_dngaussian"
  },
  {
    "method": "corgs",
    "organ": "chest",
    "views": 3,
    "psnr": 19.52733612060547,
    "ssim": 0.7242174744606018,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_chest_3views_corgs"
  },
  {
    "method": "corgs",
    "organ": "foot",
    "views": 3,
    "psnr": 25.251392364501953,
    "ssim": 0.8507952094078064,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_foot_3views_corgs"
  },
  {
    "method": "corgs",
    "organ": "head",
    "views": 3,
    "psnr": 20.537633895874023,
    "ssim": 0.8619072437286377,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_head_3views_corgs"
  },
  {
    "method": "corgs",
    "organ": "abdomen",
    "views": 3,
    "psnr": 22.588886260986328,
    "ssim": 0.9088303446769714,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_abdomen_3views_corgs"
  },
  {
    "method": "corgs",
    "organ": "pancreas",
    "views": 3,
    "psnr": 20.362464904785156,
    "ssim": 0.8799692392349243,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_pancreas_3views_corgs"
  },
  {
    "method": "corgs",
    "organ": "chest",
    "views": 6,
    "psnr": 27.6888427734375,
    "ssim": 0.8711490035057068,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_chest_6views_corgs"
  },
  {
    "method": "corgs",
    "organ": "foot",
    "views": 6,
    "psnr": 28.452329635620117,
    "ssim": 0.9006688594818115,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_foot_6views_corgs"
  },
  {
    "method": "corgs",
    "organ": "head",
    "views": 6,
    "psnr": 27.281970977783203,
    "ssim": 0.9439376592636108,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_head_6views_corgs"
  },
  {
    "method": "corgs",
    "organ": "abdomen",
    "views": 6,
    "psnr": 28.2702693939209,
    "ssim": 0.9509038329124451,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_abdomen_6views_corgs"
  },
  {
    "method": "corgs",
    "organ": "pancreas",
    "views": 6,
    "psnr": 29.28081512451172,
    "ssim": 0.9359745979309082,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_pancreas_6views_corgs"
  },
  {
    "method": "corgs",
    "organ": "chest",
    "views": 9,
    "psnr": 32.2591667175293,
    "ssim": 0.920727014541626,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_chest_9views_corgs"
  },
  {
    "method": "corgs",
    "organ": "foot",
    "views": 9,
    "psnr": 30.891128540039062,
    "ssim": 0.9281760454177856,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_08_14_foot_9views_corgs"
  },
  {
    "method": "corgs",
    "organ": "head",
    "views": 9,
    "psnr": 30.022329330444336,
    "ssim": 0.9562164545059204,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_head_9views_corgs"
  },
  {
    "method": "corgs",
    "organ": "abdomen",
    "views": 9,
    "psnr": 31.4595947265625,
    "ssim": 0.9646692872047424,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_08_50_abdomen_9views_corgs"
  },
  {
    "method": "corgs",
    "organ": "pancreas",
    "views": 9,
    "psnr": 30.775379180908203,
    "ssim": 0.9470677375793457,
    "output_path": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_pancreas_9views_corgs"
  }
]
```
