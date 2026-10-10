# CT scorecard

**Where is each organ and bone in this scan?** One label per voxel over the 117 structures the TotalSegmentator dataset marks, scored per structure by Dice overlap against the dataset's own expert masks, exactly as [METRICS_SPEC.md](METRICS_SPEC.md) defines it. Every number below was computed by `score.py` and recomputed by `recompute.py`, written from the specification alone by a worker that never saw the scorer; the two agree on all 216 cells of the test split and all 216 of development.

**The dataset publishes no patient identifier**, so the unit is the scan and two scans of one person could be treated as independent. Read every interval with that in mind.

## Ship decision

The reference is **bone-window**, the strongest baseline on development (always-empty 0.0000, bone-window 0.1847, nearest-scan 0.1550, where-it-usually-is 0.0859), chosen and recorded before the test split was opened on 2026-10-02 00:58:17.

The rule, written down before anything was trained: the segmenter ships only if the 95% interval of its mean Dice minus the reference's, on the held-out test scans, lies above zero.

- **segmenter minus bone-window: +0.5891 (95% CI +0.5728 to +0.6055)** on 89 test scans: **it ships.**

## Held-out test, opened once

Computed by `modalities/ct/eval/score.py` from prediction rows only, exactly as [METRICS_SPEC.md](METRICS_SPEC.md) defines it. 2000 resamples, seed 20261001, the unit is the scan.

**The dataset publishes no patient identifier**, so two scans of one person could be treated as independent here. Read every interval with that in mind.

### Mean Dice, by stratum

| stratum | scans | always-empty | bone-window | nearest-scan | segmenter | where-it-usually-is |
|---|---:|---:|---:|---:|---:|---:|
| all | 89 | 0.0000 (0.0000 to 0.0000) | 0.1776 (0.1591 to 0.1960) | 0.1483 (0.1313 to 0.1652) | 0.7667 (0.7517 to 0.7811) | 0.0787 (0.0658 to 0.0921) |
| abdomen | 47 | 0.0000 (0.0000 to 0.0000) | 0.2062 (0.1830 to 0.2271) | 0.1684 (0.1474 to 0.1885) | 0.7864 (0.7699 to 0.8022) | 0.1038 (0.0873 to 0.1201) |
| thorax | 32 | 0.0000 (0.0000 to 0.0000) | 0.2084 (0.1768 to 0.2365) | 0.1693 (0.1397 to 0.1974) | 0.7928 (0.7741 to 0.8086) | 0.1210 (0.0991 to 0.1410) |
| head_or_neck | 19 | 0.0000 (0.0000 to 0.0000) | 0.1833 (0.1501 to 0.2148) | 0.1578 (0.1254 to 0.1892) | 0.7674 (0.7384 to 0.7952) | 0.0847 (0.0621 to 0.1079) |
| siemens | 58 | 0.0000 (0.0000 to 0.0000) | 0.1954 (0.1745 to 0.2166) | 0.1646 (0.1455 to 0.1842) | 0.7737 (0.7523 to 0.7920) | 0.0884 (0.0701 to 0.1060) |
| not_siemens | 31 | 0.0000 (0.0000 to 0.0000) | 0.1559 (0.1224 to 0.1908) | 0.1333 (0.0992 to 0.1681) | 0.7661 (0.7433 to 0.7876) | 0.0702 (0.0538 to 0.0869) |
| main_institute | 55 | 0.0000 (0.0000 to 0.0000) | 0.1750 (0.1520 to 0.1975) | 0.1447 (0.1235 to 0.1652) | 0.7576 (0.7339 to 0.7774) | 0.0713 (0.0550 to 0.0878) |
| other_institutes | 34 | 0.0000 (0.0000 to 0.0000) | 0.1847 (0.1505 to 0.2155) | 0.1574 (0.1255 to 0.1877) | 0.7882 (0.7680 to 0.8053) | 0.0931 (0.0702 to 0.1145) |
| with_pathology | 73 | 0.0000 (0.0000 to 0.0000) | 0.1689 (0.1505 to 0.1896) | 0.1380 (0.1202 to 0.1571) | 0.7676 (0.7499 to 0.7835) | 0.0749 (0.0610 to 0.0890) |

### How often each method outlines a structure that is not there

The majority of (scan, structure) pairs are absent ones, and no Dice column would ever show this.

| method | invention rate | median invented size | structures scored |
|---|---:|---:|---:|
| always-empty | 0.0000 (0.0000 to 0.0000) | nan voxels | 115 of 117 |
| bone-window | 0.8619 (0.8370 to 0.8881) | 936 voxels | 115 of 117 |
| nearest-scan | 0.1404 (0.0809 to 0.2079) | 2469 voxels | 115 of 117 |
| segmenter | 0.0417 (0.0321 to 0.0527) | 207 voxels | 115 of 117 |
| where-it-usually-is | 1.0000 (1.0000 to 1.0000) | 5071 voxels | 115 of 117 |

### The reference

The reference is **bone-window**, the strongest baseline on the development split, chosen and written down before the test split was opened. On this split it scores 0.1776 mean Dice.

### The segmenter against the reference

Mean Dice of the segmenter minus mean Dice of bone-window, resampled on the same scans for both. The ship rule reads the first row.

| stratum | scans | segmenter | bone-window | difference (95% CI) |
|---|---:|---:|---:|---:|
| all | 89 | 0.7667 | 0.1776 | +0.5891 (+0.5728 to +0.6055) |
| abdomen | 47 | 0.7864 | 0.2062 | +0.5801 (+0.5630 to +0.5990) |
| thorax | 32 | 0.7928 | 0.2084 | +0.5844 (+0.5582 to +0.6110) |
| head_or_neck | 19 | 0.7674 | 0.1833 | +0.5840 (+0.5553 to +0.6150) |
| siemens | 58 | 0.7737 | 0.1954 | +0.5783 (+0.5576 to +0.5984) |
| not_siemens | 31 | 0.7661 | 0.1559 | +0.6103 (+0.5815 to +0.6386) |
| main_institute | 55 | 0.7576 | 0.1750 | +0.5826 (+0.5587 to +0.6062) |
| other_institutes | 34 | 0.7882 | 0.1847 | +0.6035 (+0.5806 to +0.6271) |
| with_pathology | 73 | 0.7676 | 0.1689 | +0.5987 (+0.5783 to +0.6171) |

### Every structure

Mean Dice over the scans that contain the structure, weakest first. No intervals here: the contract puts them on the headline.

| structure | segmenter | bone-window | scans containing it |
|---|---:|---:|---:|
| kidney_cyst_left | 0.0000 | 0.0000 | 4 |
| kidney_cyst_right | 0.0000 | 0.0047 | 3 |
| vertebrae_C6 | 0.5113 | 0.0628 | 34 |
| common_carotid_artery_right | 0.5467 | 0.0559 | 39 |
| vertebrae_C5 | 0.5559 | 0.0602 | 20 |
| portal_vein_and_splenic_vein | 0.5749 | 0.0629 | 65 |
| vertebrae_T7 | 0.5811 | 0.0984 | 51 |
| vertebrae_T6 | 0.5904 | 0.1213 | 44 |
| rib_right_5 | 0.6096 | 0.0289 | 57 |
| vertebrae_T8 | 0.6099 | 0.0875 | 59 |
| vertebrae_T5 | 0.6208 | 0.1439 | 43 |
| rib_right_6 | 0.6293 | 0.0225 | 64 |
| common_carotid_artery_left | 0.6323 | 0.0462 | 39 |
| vertebrae_C4 | 0.6411 | 0.0873 | 15 |
| gallbladder | 0.6426 | 0.0613 | 52 |
| adrenal_gland_left | 0.6454 | 0.0615 | 65 |
| subclavian_artery_right | 0.6466 | 0.0755 | 39 |
| adrenal_gland_right | 0.6540 | 0.0504 | 65 |
| subclavian_artery_left | 0.6587 | 0.0570 | 39 |
| vertebrae_T9 | 0.6599 | 0.1279 | 64 |
| iliac_artery_right | 0.6608 | 0.0524 | 59 |
| rib_left_5 | 0.6630 | 0.0195 | 57 |
| duodenum | 0.6660 | 0.1296 | 65 |
| rib_left_6 | 0.6740 | 0.0330 | 65 |
| rib_right_7 | 0.6771 | 0.0435 | 64 |
| iliac_artery_left | 0.6782 | 0.0535 | 59 |
| rib_right_4 | 0.6795 | 0.0368 | 48 |
| vertebrae_T4 | 0.6957 | 0.1809 | 40 |
| vertebrae_C3 | 0.6959 | 0.0936 | 14 |
| rib_right_8 | 0.6967 | 0.0424 | 66 |
| vertebrae_T10 | 0.7025 | 0.1352 | 64 |
| rib_left_3 | 0.7043 | 0.0458 | 44 |
| rib_right_3 | 0.7066 | 0.0452 | 43 |
| vertebrae_C1 | 0.7071 | 0.0892 | 22 |
| pulmonary_vein | 0.7083 | 0.0833 | 50 |
| thyroid_gland | 0.7101 | 0.1538 | 39 |
| prostate | 0.7131 | 0.2809 | 28 |
| rib_left_8 | 0.7144 | 0.0310 | 66 |
| rib_left_4 | 0.7160 | 0.0287 | 48 |
| pancreas | 0.7167 | 0.1750 | 65 |
| rib_left_7 | 0.7178 | 0.0413 | 64 |
| rib_right_9 | 0.7182 | 0.0409 | 66 |
| atrial_appendage_left | 0.7205 | 0.1029 | 40 |
| vertebrae_T3 | 0.7230 | 0.2008 | 39 |
| rib_right_10 | 0.7293 | 0.0463 | 66 |
| iliac_vena_left | 0.7306 | 0.1103 | 58 |
| vertebrae_T11 | 0.7326 | 0.1481 | 63 |
| iliac_vena_right | 0.7328 | 0.1155 | 57 |
| vertebrae_C2 | 0.7365 | 0.0887 | 21 |
| rib_right_11 | 0.7399 | 0.0433 | 66 |
| vertebrae_T12 | 0.7426 | 0.1428 | 65 |
| esophagus | 0.7443 | 0.1477 | 71 |
| vertebrae_T2 | 0.7466 | 0.2080 | 39 |
| skull | 0.7471 | 0.1811 | 34 |
| rib_left_9 | 0.7477 | 0.0408 | 66 |
| vertebrae_C7 | 0.7486 | 0.2121 | 37 |
| costal_cartilages | 0.7517 | 0.0087 | 72 |
| rib_left_10 | 0.7556 | 0.0332 | 66 |
| rib_left_11 | 0.7578 | 0.0319 | 66 |
| vertebrae_L1 | 0.7598 | 0.1487 | 64 |
| small_bowel | 0.7612 | 0.2321 | 64 |
| rib_left_12 | 0.7638 | 0.0355 | 62 |
| humerus_right | 0.7676 | 0.1311 | 41 |
| colon | 0.7687 | 0.1194 | 69 |
| rib_right_12 | 0.7697 | 0.0388 | 62 |
| brachiocephalic_vein_right | 0.7724 | 0.1132 | 40 |
| inferior_vena_cava | 0.7784 | 0.2222 | 72 |
| vertebrae_T1 | 0.7799 | 0.2134 | 38 |
| superior_vena_cava | 0.7838 | 0.1422 | 47 |
| rib_left_2 | 0.7866 | 0.0820 | 41 |
| vertebrae_L2 | 0.7898 | 0.1677 | 59 |
| brachiocephalic_vein_left | 0.7943 | 0.1047 | 40 |
| urinary_bladder | 0.7962 | 0.3546 | 54 |
| lung_upper_lobe_right | 0.8004 | 0.5295 | 54 |
| sternum | 0.8038 | 0.1228 | 68 |
| brachiocephalic_trunk | 0.8048 | 0.0837 | 39 |
| vertebrae_L3 | 0.8054 | 0.1736 | 59 |
| rib_left_1 | 0.8098 | 0.1361 | 39 |
| rib_right_2 | 0.8145 | 0.0780 | 40 |
| vertebrae_L4 | 0.8192 | 0.1775 | 60 |
| rib_right_1 | 0.8314 | 0.1595 | 39 |
| sacrum | 0.8380 | 0.1629 | 57 |
| humerus_left | 0.8413 | 0.1460 | 40 |
| scapula_left | 0.8413 | 0.1651 | 48 |
| gluteus_minimus_left | 0.8432 | 0.2034 | 54 |
| iliopsoas_left | 0.8437 | 0.2993 | 69 |
| stomach | 0.8455 | 0.2723 | 66 |
| vertebrae_S1 | 0.8480 | 0.2208 | 55 |
| vertebrae_L5 | 0.8523 | 0.1884 | 57 |
| gluteus_minimus_right | 0.8532 | 0.2208 | 53 |
| scapula_right | 0.8569 | 0.1846 | 48 |
| lung_middle_lobe_right | 0.8584 | 0.4242 | 64 |
| heart | 0.8680 | 0.4046 | 69 |
| kidney_right | 0.8691 | 0.2962 | 64 |
| clavicula_left | 0.8699 | 0.1775 | 40 |
| spinal_cord | 0.8719 | 0.1709 | 88 |
| iliopsoas_right | 0.8751 | 0.3159 | 67 |
| kidney_left | 0.8757 | 0.2978 | 68 |
| aorta | 0.8830 | 0.2421 | 77 |
| brain | 0.8836 | 0.5128 | 22 |
| trachea | 0.8837 | 0.3039 | 43 |
| clavicula_right | 0.8943 | 0.1762 | 39 |
| lung_upper_lobe_left | 0.8985 | 0.5366 | 71 |
| gluteus_medius_right | 0.9031 | 0.3603 | 55 |
| lung_lower_lobe_left | 0.9047 | 0.5299 | 72 |
| lung_lower_lobe_right | 0.9055 | 0.5449 | 71 |
| gluteus_medius_left | 0.9060 | 0.3815 | 57 |
| autochthon_right | 0.9142 | 0.4381 | 81 |
| femur_right | 0.9157 | 0.3610 | 54 |
| femur_left | 0.9206 | 0.3104 | 55 |
| hip_right | 0.9239 | 0.2997 | 59 |
| autochthon_left | 0.9240 | 0.4456 | 81 |
| liver | 0.9258 | 0.5619 | 70 |
| gluteus_maximus_right | 0.9319 | 0.5179 | 54 |
| gluteus_maximus_left | 0.9363 | 0.5142 | 54 |
| hip_left | 0.9400 | 0.3091 | 58 |
| spleen | 0.9427 | 0.3502 | 65 |



## Development, each scan scored by a model that never saw it

Computed by `modalities/ct/eval/score.py` from prediction rows only, exactly as [METRICS_SPEC.md](METRICS_SPEC.md) defines it. 2000 resamples, seed 20261001, the unit is the scan.

**The dataset publishes no patient identifier**, so two scans of one person could be treated as independent here. Read every interval with that in mind.

### Mean Dice, by stratum

| stratum | scans | always-empty | bone-window | nearest-scan | segmenter | where-it-usually-is |
|---|---:|---:|---:|---:|---:|---:|
| all | 1139 | 0.0000 (0.0000 to 0.0000) | 0.1847 (0.1798 to 0.1894) | 0.1550 (0.1506 to 0.1594) | 0.7255 (0.7212 to 0.7295) | 0.0859 (0.0821 to 0.0900) |
| abdomen | 455 | 0.0000 (0.0000 to 0.0000) | 0.2044 (0.1974 to 0.2116) | 0.1645 (0.1583 to 0.1707) | 0.7352 (0.7269 to 0.7422) | 0.1059 (0.1008 to 0.1112) |
| thorax | 442 | 0.0000 (0.0000 to 0.0000) | 0.2148 (0.2080 to 0.2217) | 0.1741 (0.1678 to 0.1804) | 0.7409 (0.7353 to 0.7465) | 0.1212 (0.1157 to 0.1264) |
| head_or_neck | 295 | 0.0000 (0.0000 to 0.0000) | 0.2175 (0.2081 to 0.2273) | 0.1773 (0.1682 to 0.1867) | 0.7510 (0.7432 to 0.7584) | 0.1163 (0.1084 to 0.1241) |
| siemens | 1027 | 0.0000 (0.0000 to 0.0000) | 0.1850 (0.1797 to 0.1902) | 0.1555 (0.1506 to 0.1600) | 0.7242 (0.7195 to 0.7289) | 0.0856 (0.0813 to 0.0900) |
| not_siemens | 112 | 0.0000 (0.0000 to 0.0000) | 0.1863 (0.1736 to 0.1989) | 0.1545 (0.1428 to 0.1657) | 0.7495 (0.7377 to 0.7602) | 0.0913 (0.0804 to 0.1021) |
| main_institute | 843 | 0.0000 (0.0000 to 0.0000) | 0.1820 (0.1761 to 0.1879) | 0.1536 (0.1484 to 0.1589) | 0.7213 (0.7162 to 0.7263) | 0.0830 (0.0782 to 0.0878) |
| other_institutes | 296 | 0.0000 (0.0000 to 0.0000) | 0.1928 (0.1845 to 0.2011) | 0.1592 (0.1513 to 0.1672) | 0.7410 (0.7332 to 0.7489) | 0.0950 (0.0887 to 0.1015) |
| with_pathology | 751 | 0.0000 (0.0000 to 0.0000) | 0.1815 (0.1756 to 0.1871) | 0.1528 (0.1473 to 0.1582) | 0.7236 (0.7181 to 0.7289) | 0.0848 (0.0803 to 0.0894) |

### How often each method outlines a structure that is not there

The majority of (scan, structure) pairs are absent ones, and no Dice column would ever show this.

| method | invention rate | median invented size | structures scored |
|---|---:|---:|---:|
| always-empty | 0.0000 (0.0000 to 0.0000) | nan voxels | 117 of 117 |
| bone-window | 0.8719 (0.8626 to 0.8808) | 804 voxels | 117 of 117 |
| nearest-scan | 0.1124 (0.1003 to 0.1249) | 1738 voxels | 117 of 117 |
| segmenter | 0.1414 (0.1352 to 0.1481) | 42 voxels | 117 of 117 |
| where-it-usually-is | 1.0000 (1.0000 to 1.0000) | 4048 voxels | 117 of 117 |

### The reference

The reference is **bone-window**, the strongest baseline on the development split, chosen and written down before the test split was opened. On this split it scores 0.1847 mean Dice.

### The segmenter against the reference

Mean Dice of the segmenter minus mean Dice of bone-window, resampled on the same scans for both. The ship rule reads the first row.

| stratum | scans | segmenter | bone-window | difference (95% CI) |
|---|---:|---:|---:|---:|
| all | 1139 | 0.7255 | 0.1847 | +0.5408 (+0.5362 to +0.5454) |
| abdomen | 455 | 0.7352 | 0.2044 | +0.5307 (+0.5226 to +0.5379) |
| thorax | 442 | 0.7409 | 0.2148 | +0.5261 (+0.5198 to +0.5326) |
| head_or_neck | 295 | 0.7510 | 0.2175 | +0.5335 (+0.5239 to +0.5427) |
| siemens | 1027 | 0.7242 | 0.1850 | +0.5393 (+0.5342 to +0.5443) |
| not_siemens | 112 | 0.7495 | 0.1863 | +0.5633 (+0.5499 to +0.5758) |
| main_institute | 843 | 0.7213 | 0.1820 | +0.5394 (+0.5340 to +0.5450) |
| other_institutes | 296 | 0.7410 | 0.1928 | +0.5483 (+0.5403 to +0.5562) |
| with_pathology | 751 | 0.7236 | 0.1815 | +0.5421 (+0.5365 to +0.5477) |

### Every structure

Mean Dice over the scans that contain the structure, weakest first. No intervals here: the contract puts them on the headline.

| structure | segmenter | bone-window | scans containing it |
|---|---:|---:|---:|
| kidney_cyst_left | 0.0000 | 0.0146 | 26 |
| kidney_cyst_right | 0.0000 | 0.0059 | 22 |
| common_carotid_artery_right | 0.4679 | 0.0362 | 682 |
| vertebrae_C5 | 0.5141 | 0.0945 | 329 |
| portal_vein_and_splenic_vein | 0.5543 | 0.0561 | 765 |
| vertebrae_C6 | 0.5664 | 0.1296 | 522 |
| vertebrae_T7 | 0.5793 | 0.1547 | 694 |
| vertebrae_T6 | 0.5898 | 0.1568 | 688 |
| vertebrae_T8 | 0.5937 | 0.1511 | 737 |
| common_carotid_artery_left | 0.5954 | 0.0473 | 714 |
| duodenum | 0.6001 | 0.1195 | 703 |
| rib_right_5 | 0.6049 | 0.0480 | 783 |
| rib_right_6 | 0.6050 | 0.0521 | 805 |
| prostate | 0.6121 | 0.2313 | 317 |
| subclavian_artery_right | 0.6133 | 0.0737 | 692 |
| vertebrae_T9 | 0.6135 | 0.1564 | 785 |
| adrenal_gland_left | 0.6197 | 0.0468 | 735 |
| rib_left_5 | 0.6198 | 0.0427 | 811 |
| subclavian_artery_left | 0.6287 | 0.0674 | 712 |
| vertebrae_T5 | 0.6304 | 0.1704 | 690 |
| rib_left_6 | 0.6328 | 0.0445 | 844 |
| adrenal_gland_right | 0.6407 | 0.0431 | 762 |
| rib_right_7 | 0.6426 | 0.0531 | 787 |
| vertebrae_T10 | 0.6434 | 0.1720 | 813 |
| vertebrae_C4 | 0.6444 | 0.1254 | 235 |
| rib_right_4 | 0.6451 | 0.0483 | 739 |
| iliac_artery_right | 0.6467 | 0.0639 | 585 |
| rib_right_8 | 0.6511 | 0.0482 | 794 |
| rib_left_8 | 0.6517 | 0.0534 | 822 |
| rib_left_4 | 0.6524 | 0.0472 | 758 |
| rib_left_7 | 0.6528 | 0.0505 | 827 |
| iliac_artery_left | 0.6530 | 0.0576 | 592 |
| gallbladder | 0.6539 | 0.0533 | 603 |
| thyroid_gland | 0.6541 | 0.1293 | 665 |
| vertebrae_T4 | 0.6621 | 0.1787 | 698 |
| rib_left_9 | 0.6653 | 0.0531 | 824 |
| vertebrae_T11 | 0.6709 | 0.1749 | 824 |
| vertebrae_C3 | 0.6735 | 0.1362 | 215 |
| rib_left_12 | 0.6798 | 0.0257 | 728 |
| pancreas | 0.6817 | 0.1402 | 746 |
| rib_right_9 | 0.6846 | 0.0538 | 794 |
| vertebrae_T3 | 0.6856 | 0.1829 | 703 |
| vertebrae_T12 | 0.6885 | 0.1724 | 806 |
| rib_left_3 | 0.6899 | 0.0526 | 729 |
| vertebrae_C7 | 0.6902 | 0.2331 | 654 |
| rib_left_10 | 0.6927 | 0.0495 | 814 |
| rib_right_11 | 0.6933 | 0.0445 | 800 |
| rib_right_3 | 0.6946 | 0.0519 | 723 |
| rib_right_12 | 0.6967 | 0.0304 | 723 |
| vertebrae_C1 | 0.6983 | 0.1452 | 228 |
| rib_right_10 | 0.7003 | 0.0502 | 800 |
| iliac_vena_right | 0.7022 | 0.1312 | 572 |
| vertebrae_L2 | 0.7051 | 0.1769 | 653 |
| iliac_vena_left | 0.7081 | 0.1255 | 578 |
| pulmonary_vein | 0.7088 | 0.0912 | 743 |
| vertebrae_L1 | 0.7092 | 0.1693 | 737 |
| esophagus | 0.7107 | 0.1484 | 962 |
| rib_left_11 | 0.7112 | 0.0364 | 791 |
| small_bowel | 0.7150 | 0.2100 | 747 |
| costal_cartilages | 0.7161 | 0.0113 | 945 |
| vertebrae_T2 | 0.7167 | 0.2036 | 685 |
| vertebrae_L3 | 0.7201 | 0.1875 | 586 |
| atrial_appendage_left | 0.7388 | 0.0822 | 633 |
| vertebrae_T1 | 0.7388 | 0.2308 | 666 |
| colon | 0.7388 | 0.1152 | 844 |
| vertebrae_C2 | 0.7410 | 0.1785 | 232 |
| brachiocephalic_trunk | 0.7465 | 0.0697 | 692 |
| vertebrae_L4 | 0.7479 | 0.1987 | 576 |
| brachiocephalic_vein_right | 0.7537 | 0.1041 | 695 |
| brachiocephalic_vein_left | 0.7568 | 0.1109 | 691 |
| skull | 0.7650 | 0.1919 | 439 |
| rib_left_2 | 0.7654 | 0.0812 | 711 |
| vertebrae_L5 | 0.7666 | 0.2026 | 562 |
| rib_right_2 | 0.7674 | 0.0766 | 711 |
| inferior_vena_cava | 0.7695 | 0.2453 | 905 |
| rib_left_1 | 0.7719 | 0.1205 | 703 |
| rib_right_1 | 0.7760 | 0.1249 | 699 |
| vertebrae_S1 | 0.7855 | 0.2105 | 542 |
| urinary_bladder | 0.7984 | 0.3625 | 517 |
| superior_vena_cava | 0.8051 | 0.1656 | 730 |
| gluteus_minimus_left | 0.8070 | 0.2180 | 524 |
| sternum | 0.8141 | 0.1372 | 891 |
| gluteus_minimus_right | 0.8186 | 0.2462 | 516 |
| humerus_right | 0.8198 | 0.1772 | 577 |
| stomach | 0.8203 | 0.2564 | 839 |
| sacrum | 0.8218 | 0.1574 | 561 |
| iliopsoas_right | 0.8233 | 0.2996 | 775 |
| humerus_left | 0.8235 | 0.1759 | 574 |
| iliopsoas_left | 0.8257 | 0.2893 | 767 |
| scapula_right | 0.8315 | 0.1969 | 738 |
| scapula_left | 0.8330 | 0.1898 | 761 |
| lung_upper_lobe_right | 0.8388 | 0.5992 | 840 |
| brain | 0.8420 | 0.5793 | 227 |
| clavicula_right | 0.8458 | 0.1572 | 684 |
| lung_middle_lobe_right | 0.8501 | 0.4693 | 817 |
| spinal_cord | 0.8514 | 0.1945 | 1095 |
| clavicula_left | 0.8565 | 0.1419 | 687 |
| trachea | 0.8606 | 0.2942 | 749 |
| gluteus_medius_left | 0.8681 | 0.3837 | 568 |
| kidney_left | 0.8720 | 0.2884 | 734 |
| gluteus_medius_right | 0.8772 | 0.4043 | 558 |
| aorta | 0.8782 | 0.2312 | 1027 |
| lung_lower_lobe_right | 0.8784 | 0.5818 | 920 |
| lung_lower_lobe_left | 0.8803 | 0.5580 | 934 |
| kidney_right | 0.8812 | 0.2691 | 718 |
| heart | 0.8908 | 0.4347 | 855 |
| autochthon_left | 0.8946 | 0.4239 | 1052 |
| autochthon_right | 0.8976 | 0.4233 | 1050 |
| spleen | 0.8986 | 0.3397 | 821 |
| gluteus_maximus_right | 0.9031 | 0.5350 | 540 |
| gluteus_maximus_left | 0.9037 | 0.5202 | 541 |
| lung_upper_lobe_left | 0.9067 | 0.5992 | 953 |
| hip_left | 0.9110 | 0.2952 | 588 |
| femur_right | 0.9120 | 0.3379 | 503 |
| hip_right | 0.9148 | 0.3042 | 580 |
| femur_left | 0.9153 | 0.3211 | 511 |
| liver | 0.9397 | 0.5931 | 869 |



---

Not a medical device. These are measurements on a public research dataset, not a claim about anybody's scan.

## Second look, version 116 (look 2 of the test)

This is look 2 of the CT test split. The first look (version 1, opened 2026-10-02 00:58:17) is above, unchanged; this one is labelled a second look, and the result is reported whether it is higher or lower.

**Development gate**, passed before the test was touched: version 116 minus version 1 on 1139 development scans, paired, per-scan mean Dice: +0.0650 (95% CI +0.0630 to +0.0670, 2000 resamples). Version 116: 0.7920, version 1: 0.7269.

The reference is still **bone-window**, fixed at the first look.

| | test mean Dice | minus bone-window |
|---|---|---|
| version 116 (look 2) | 0.8318 (95% CI 0.8191 to 0.8425) | +0.6542 (95% CI +0.6380 to +0.6711) |
| version 1 (look 1, from its scorecard) | 0.7667 (95% CI 0.7517 to 0.7811) | +0.5891 (95% CI +0.5728 to +0.6055) |

The independent recomputation (`recompute.py`) agrees on all 216 test cells and 216 development cells of this look.

Files: `eval/ct/look2-v116`.

## Second look, version 116 with version 115 (look 3 of the test)

This is look 3 of the CT test split. The first look (version 1, opened 2026-10-02 00:58:17) is above, unchanged; this one is labelled a second look, and the result is reported whether it is higher or lower.

This read is a **pair**: version 116's five nets and version 115's five, read together with every net averaged with equal weight (ten nets), under version 116's reading. Its development rows are version 116's out-of-fold files tagged `pair115`.

**Development gate**, passed before the test was touched: version 116 with version 115 minus version 116 on 1139 development scans, paired, per-scan mean Dice: +0.0098 (95% CI +0.0091 to +0.0105, 2000 resamples), passing only with a low end above noise 0.0037. version 116 with version 115: 0.8017. version 116: 0.7920.

The reference is still **bone-window**, fixed at the first look.

| | test mean Dice | minus bone-window |
|---|---|---|
| version 116 with version 115 (look 3) | 0.8336 (95% CI 0.8214 to 0.8442) | +0.6560 (95% CI +0.6400 to +0.6729) |
| version 1 (look 1, from its scorecard) | 0.7667 (95% CI 0.7517 to 0.7811) | +0.5891 (95% CI +0.5728 to +0.6055) |

The independent recomputation (`recompute.py`) agrees on all 216 test cells and 216 development cells of this look.

Files: `eval/ct/look3-v116+115`.

## Second look, version 121 (look 4 of the test)

This is look 4 of the CT test split. The first look (version 1, opened 2026-10-02 00:58:17) is above, unchanged; this one is labelled a second look, and the result is reported whether it is higher or lower.

This read is at **1.5 mm**: each test scan was read from its own 1.5 mm cache, with version 116's 3 mm coarse map as the prior.

**Development gate**, passed before the test was touched: version 121 minus version 116 with version 115 on 1139 development scans, paired, per-scan mean Dice: +0.0153 (95% CI +0.0140 to +0.0168, 2000 resamples), passing only with a low end above noise 0.0037. version 121: 0.8171. version 116 with version 115: 0.8017.

The reference is still **bone-window**, fixed at the first look.

| | test mean Dice | minus bone-window |
|---|---|---|
| version 121 (look 4) | 0.8668 (95% CI 0.8538 to 0.8781) | +0.6892 (95% CI +0.6728 to +0.7062) |
| version 1 (look 1, from its scorecard) | 0.7667 (95% CI 0.7517 to 0.7811) | +0.5891 (95% CI +0.5728 to +0.6055) |

The independent recomputation (`recompute.py`) agrees on all 216 test cells and 216 development cells of this look.

Files: `eval/ct/look4-v121`.
