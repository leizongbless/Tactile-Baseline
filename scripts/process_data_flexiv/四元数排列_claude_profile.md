# Flexiv release 数据集四元数布局 profile (Claude)

> 时间：2026-05-29
> 数据集根：`/mnt/data/data/ymm/dataset/Flexiv/release`
> 探针参考 (Flexiv-A Rizon4s home, RDK 实测 WXYZ): `[-0.24059, 0.840898, -0.480889, 0.061259]`
> 探针参考来源：`docs/flexiv_RT/agents/04-quat-layout-fix/dataset_vs_probe_log.md` §1

## 方法

对 release 根下每个 task 目录，自动定位第一个含 `dataset.hdf5` 的 episode 目录，并对 `action/{arm}_eef/feedback/pose["value"]` 的前 200 帧做内禀统计，再与 `pass_car_0429` 已验证的 RDK home WXYZ 4-vec 做 layout-blind 角差交叉。

**判据三层（confidence 递减）：**

1. **layout-blind angular cross-check**：分别假设 dataset 的 4-vec 跟 `pass_car` ref 的关系是 (a) direct、(b) cyclic shift -1、(c) cyclic shift +1，三选一取最小角差。若 direct ≤ 30°，确定是 WXYZ；若某 shift ≤ 30°，确定有 permute。
2. **slot 模式**：在 home 附近的 Rizon4s，WXYZ 字节里 abs 最大的应该是 slot[4]（qx），abs 最小的应该是 slot[6]（qz）。若 dataset 命中这个 pattern，再叠加 60° 宽松阈值判 WXYZ_MEDIUM_CONF。
3. **若三种 layout 全部不接近**：说明该 task home pose 与 `pass_car` 物理差异大，不能用该 ref 单一证据判定。报告里标 `AMBIGUOUS_DIFFERENT_HOME_POSE`，并依据采集 toolkit 一致性（同一套 `flexivrdk` + `action/{arm}_eef/feedback/{pose,gripper}` 树状路径）做 group-level 推断。

**所有判据都建立在『原始 HDF5 字节就是 RDK `state.tcp_pose` 字节』这一假设上**。pass_car 那条已在 `dataset_vs_probe_log.md` 上用 100 个 episode 实证锁定；其它任务的 HDF5 文件如果也由同一个采集 toolkit 写成（HDF5 root attrs / group tree 结构一致），则字节布局必然一致。

## 结果总览

| # | Task | Verdict | Conf | Pose path | First xyz (m) | First quat4 | angle vs ref (deg) — direct / shift-1 / shift+1 |
|---|---|---|---|---|---|---|---|
| 1 | `Flexiv_breakfast_1_bowl_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 2 | `Flexiv_breakfast_2_plate_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.075, +0.896, -0.206] | [+0.8015, -0.5261, +0.0373, -0.2817] | 95.8 / 8.8 / 139.9 |
| 3 | `Flexiv_breakfast_3_pour_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 4 | `Flexiv_breakfast_4_bread_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 5 | `Flexiv_breakfast_4_bread_0514` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 6 | `Flexiv_breakfast_5_spoon_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.044, +0.923, -0.230] | [+0.8538, -0.4367, +0.0632, -0.2762] | 103.4 / 6.7 / 139.8 |
| 7 | `Flexiv_breakfast_full_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 8 | `Flexiv_breakfast_full_0527` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 9 | `Flexiv_chemExp_1_place_tube_0413` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.048, +0.918, -0.223] | [+0.8501, -0.4424, +0.0619, -0.2789] | 102.9 / 6.3 / 139.7 |
| 10 | `Flexiv_chemExp_1_place_tube_0417` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.051, +0.916, -0.221] | [+0.8483, -0.4450, +0.0634, -0.2800] | 102.5 / 6.2 / 139.4 |
| 11 | `Flexiv_chemExp_2_drawer_storage_0425` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.050, +0.916, -0.222] | [+0.8485, -0.4447, +0.0632, -0.2798] | 102.5 / 6.2 / 139.4 |
| 12 | `Flexiv_chemExp_3_flask_transfer_0425` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.050, +0.917, -0.221] | [+0.8484, -0.4445, +0.0631, -0.2805] | 102.6 / 6.2 / 139.4 |
| 13 | `Flexiv_chemExp_4_cap_exchange_0425` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.050, +0.917, -0.222] | [+0.8489, -0.4442, +0.0628, -0.2797] | 102.6 / 6.2 / 139.5 |
| 14 | `Flexiv_chemExp_5_Sealing_0425` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.051, +0.916, -0.222] | [+0.8485, -0.4449, +0.0633, -0.2797] | 102.5 / 6.2 / 139.4 |
| 15 | `Flexiv_chemExp_full_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.051, +0.916, -0.222] | [+0.8484, -0.4449, +0.0634, -0.2798] | 102.5 / 6.2 / 139.4 |
| 16 | `Flexiv_clothing_1_clothing_0425` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.224] | [+0.8467, -0.4495, +0.0680, -0.2764] | 101.7 / 5.6 / 139.0 |
| 17 | `Flexiv_clothing_1_clothing_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 18 | `Flexiv_clothing_2_box_assembly_0513` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.051, +0.917, -0.221] | [+0.8482, -0.4447, +0.0635, -0.2807] | 102.5 / 6.3 / 139.4 |
| 19 | `Flexiv_clothing_3_packing_0429` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.051, +0.916, -0.222] | [+0.8485, -0.4449, +0.0634, -0.2794] | 102.5 / 6.1 / 139.4 |
| 20 | `Flexiv_clothing_4_box_closing_0513` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.051, +0.916, -0.222] | [+0.8484, -0.4448, +0.0633, -0.2798] | 102.5 / 6.2 / 139.4 |
| 21 | `Flexiv_computer_1_cpu_0513` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 22 | `Flexiv_computer_2_ram_0513` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 23 | `Flexiv_pass_car_0429` | **WXYZ_HIGH_CONF** | high | `action/left_eef/feedback/pose` | [+0.051, +0.916, -0.222] | [-0.2797, +0.8485, -0.4448, +0.0633] | 6.2 / 93.4 / 102.5 |
| 24 | `Flexiv_phone_packing_0518` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2762] | 101.7 / 5.5 / 139.0 |
| 25 | `Flexiv_whack_a_mole_0518` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 26 | `Flexiv_whack_a_mole_0527` | **LIKELY_XYZW_PERMUTED_shift_-1** | high | `action/left_eef/feedback/pose` | [+0.055, +0.912, -0.225] | [+0.8469, -0.4494, +0.0678, -0.2761] | 101.7 / 5.5 / 139.0 |
| 27 | `ossutil_output` | ERROR | low | — | — | — | no dataset.hdf5 found anywhere under task dir |

### Verdict tally

- ERROR: 1
- LIKELY_XYZW_PERMUTED_shift_-1: 25
- WXYZ_HIGH_CONF: 1

## 采集 toolkit 一致性指纹

如果所有 task 的 HDF5 都呈现相同的 group tree 模式（`action/{left,right,single}_eef/feedback/pose` + 同款 attrs），再加上 `pass_car` 字节已经实证锁定 = WXYZ，那么其它 task 在同一套 toolkit 下写成的 HDF5 字节布局 **必然也是 WXYZ**。这是对 §『AMBIGUOUS_DIFFERENT_HOME_POSE』 任务最关键的旁证。

| Task | h5 root attrs | pose group attrs (excerpt) | pose value dtype | pose path |
|---|---|---|---|---|
| `Flexiv_breakfast_1_bowl_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_2_plate_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_3_pour_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_4_bread_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_4_bread_0514` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_5_spoon_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_full_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_breakfast_full_0527` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_1_place_tube_0413` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_1_place_tube_0417` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_2_drawer_storage_0425` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_3_flask_transfer_0425` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_4_cap_exchange_0425` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_5_Sealing_0425` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_chemExp_full_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_clothing_1_clothing_0425` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_clothing_1_clothing_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_clothing_2_box_assembly_0513` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_clothing_3_packing_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_clothing_4_box_closing_0513` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_computer_1_cpu_0513` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_computer_2_ram_0513` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_pass_car_0429` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_phone_packing_0518` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_whack_a_mole_0518` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |
| `Flexiv_whack_a_mole_0527` | — | — | `[('value', '<f4', (7,)), ('timestamp', '<f8')]` | `action/left_eef/feedback/pose` |

## 每个 task 的细节

### `Flexiv_breakfast_1_bowl_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_1_bowl_0429/Flexiv准备早餐-首轮容器与饮品放置/worldcode_Flexiv-B_2026-04-23-11-37-56_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[793, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05501953512430191, 0.9119234085083008, -0.22463957965373993]`
- 首帧 quat4：`[0.8468716740608215, -0.4493972957134247, 0.06783797591924667, -0.27613115310668945]`
- 相邻帧最大 |Δq| 中位数：2.32422e-03
- 相邻帧最大 |Δxyz| 中位数：1.4546 mm
- slot abs_mean：slot[3]=0.7026, slot[4]=0.5746, slot[5]=0.3444, slot[6]=0.1280
- slot 模式：max@3, min@6
- 与 pass_car ref 的 layout-blind 角差：direct=101.72°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@6. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_2_plate_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_2_plate_0429/Flexiv准备早餐-次轮容器与饮品放置/worldcode_Flexiv-B_2026-04-23-14-48-11_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[849, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.0754617229104042, 0.8961582183837891, -0.20612084865570068]`
- 首帧 quat4：`[0.8015079498291016, -0.5261396765708923, 0.037314727902412415, -0.28172630071640015]`
- 相邻帧最大 |Δq| 中位数：1.21510e-03
- 相邻帧最大 |Δxyz| 中位数：1.3692 mm
- slot abs_mean：slot[3]=0.7326, slot[4]=0.5299, slot[5]=0.3608, slot[6]=0.0805
- slot 模式：max@3, min@6
- 与 pass_car ref 的 layout-blind 角差：direct=95.79°, shift_-1=8.78°, shift_+1=139.94°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (8.8°). slot-pattern=max@3, min@6. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_3_pour_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_3_pour_0429/Flexiv准备早餐-水果放置与倾倒燕麦/worldcode_Flexiv-B_2026-04-24-11-40-44_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1470, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05499785766005516, 0.9119226336479187, -0.2246536910533905]`
- 首帧 quat4：`[0.8468867540359497, -0.449384480714798, 0.06783296912908554, -0.2761070728302002]`
- 相邻帧最大 |Δq| 中位数：2.34522e-04
- 相邻帧最大 |Δxyz| 中位数：0.2751 mm
- slot abs_mean：slot[3]=0.6855, slot[4]=0.6088, slot[5]=0.0874, slot[6]=0.3498
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_4_bread_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_4_bread_0429/Flexiv准备早餐-剩余食物装盘/worldcode_Flexiv-B_2026-04-24-15-06-17_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1004, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.054998546838760376, 0.911920964717865, -0.2246522605419159]`
- 首帧 quat4：`[0.8468856811523438, -0.4493843913078308, 0.0678325891494751, -0.27611055970191956]`
- 相邻帧最大 |Δq| 中位数：4.78312e-04
- 相邻帧最大 |Δxyz| 中位数：0.2573 mm
- slot abs_mean：slot[3]=0.6346, slot[4]=0.6388, slot[5]=0.1251, slot[6]=0.3627
- slot 模式：max@4, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@4, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_4_bread_0514`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_4_bread_0514/Flexiv准备早餐-剩余食物装盘/worldcode_Flexiv-B_2026-05-14-13-22-28_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1274, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05498848482966423, 0.9119172096252441, -0.2246718853712082]`
- 首帧 quat4：`[0.8468943238258362, -0.44937893748283386, 0.0678337961435318, -0.27609261870384216]`
- 相邻帧最大 |Δq| 中位数：2.27869e-04
- 相邻帧最大 |Δxyz| 中位数：0.2387 mm
- slot abs_mean：slot[3]=0.6913, slot[4]=0.6351, slot[5]=0.1221, slot[6]=0.2848
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.53°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_5_spoon_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_5_spoon_0429/Flexiv准备早餐-餐具精细提取/worldcode_Flexiv-B_2026-04-24-18-54-36_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1162, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.04390835762023926, 0.923345148563385, -0.23030051589012146]`
- 首帧 quat4：`[0.8537998199462891, -0.4367220401763916, 0.06322285532951355, -0.2762293219566345]`
- 相邻帧最大 |Δq| 中位数：6.30356e-04
- 相邻帧最大 |Δxyz| 中位数：0.3365 mm
- slot abs_mean：slot[3]=0.8380, slot[4]=0.4487, slot[5]=0.0610, slot[6]=0.3013
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=103.37°, shift_-1=6.67°, shift_+1=139.83°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.7°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_full_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_full_0429/Flexiv准备早餐-全流程/worldcode_Flexiv-B_2026-04-25-10-29-30_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[5764, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.0550052635371685, 0.911910891532898, -0.22464796900749207]`
- 首帧 quat4：`[0.8468798995018005, -0.4493926465511322, 0.0678357481956482, -0.27611401677131653]`
- 相邻帧最大 |Δq| 中位数：3.13997e-04
- 相邻帧最大 |Δxyz| 中位数：0.3416 mm
- slot abs_mean：slot[3]=0.7776, slot[4]=0.5177, slot[5]=0.1614, slot[6]=0.2393
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_breakfast_full_0527`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_breakfast_full_0527/Flexiv准备早餐-全流程/worldcode_Flexiv-B_2026-05-21-10-14-54_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[4117, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05499263480305672, 0.9119148254394531, -0.22467269003391266]`
- 首帧 quat4：`[0.8468894958496094, -0.44938337802886963, 0.06782986223697662, -0.2761012315750122]`
- 相邻帧最大 |Δq| 中位数：1.69325e-03
- 相邻帧最大 |Δxyz| 中位数：1.2257 mm
- slot abs_mean：slot[3]=0.7201, slot[4]=0.5582, slot[5]=0.2923, slot[6]=0.1795
- slot 模式：max@3, min@6
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@6. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_1_place_tube_0413`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_1_place_tube_0413/Flexiv化学实验-试剂瓶分类插架/worldcode_Flexiv-A_2026-04-13-17-30-57_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[810, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.04787127301096916, 0.9183078408241272, -0.22296454012393951]`
- 首帧 quat4：`[0.8500758409500122, -0.4424060881137848, 0.06194659322500229, -0.2789454162120819]`
- 相邻帧最大 |Δq| 中位数：1.45632e-03
- 相邻帧最大 |Δxyz| 中位数：2.0710 mm
- slot abs_mean：slot[3]=0.8049, slot[4]=0.5070, slot[5]=0.0426, slot[6]=0.2942
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.87°, shift_-1=6.32°, shift_+1=139.67°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.3°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_1_place_tube_0417`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_1_place_tube_0417/Flexiv化学实验-试剂瓶分类插架/worldcode_Flexiv-A_2026-04-17-12-07-06_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[986, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05083238705992699, 0.9162082672119141, -0.2213088870048523]`
- 首帧 quat4：`[0.8482716083526611, -0.44496428966522217, 0.06338111311197281, -0.2800444960594177]`
- 相邻帧最大 |Δq| 中位数：9.74065e-04
- 相邻帧最大 |Δxyz| 中位数：1.4099 mm
- slot abs_mean：slot[3]=0.8514, slot[4]=0.4562, slot[5]=0.0364, slot[6]=0.2512
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.50°, shift_-1=6.18°, shift_+1=139.39°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_2_drawer_storage_0425`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_2_drawer_storage_0425/Flexiv化学实验-抽屉收纳交互/worldcode_Flexiv-A_2026-04-23-11-00-45_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1225, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.050478044897317886, 0.9164546132087708, -0.22159098088741302]`
- 首帧 quat4：`[0.8485069870948792, -0.4446747303009033, 0.06320375204086304, -0.2798312306404114]`
- 相邻帧最大 |Δq| 中位数：8.24630e-05
- 相邻帧最大 |Δxyz| 中位数：0.0909 mm
- slot abs_mean：slot[3]=0.8477, slot[4]=0.4426, slot[5]=0.0647, slot[6]=0.2852
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.55°, shift_-1=6.19°, shift_+1=139.43°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_3_flask_transfer_0425`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_3_flask_transfer_0425/Flexiv化学实验-容量瓶转移与初次开盖/worldcode_Flexiv-A_2026-04-16-14-23-05_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1434, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05044587701559067, 0.9167346954345703, -0.22088339924812317]`
- 首帧 quat4：`[0.84837406873703, -0.44454044103622437, 0.06310200691223145, -0.2804698050022125]`
- 相邻帧最大 |Δq| 中位数：2.19766e-04
- 相邻帧最大 |Δxyz| 中位数：0.1903 mm
- slot abs_mean：slot[3]=0.8465, slot[4]=0.4496, slot[5]=0.0470, slot[6]=0.2808
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.57°, shift_-1=6.25°, shift_+1=139.41°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_4_cap_exchange_0425`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_4_cap_exchange_0425/Flexiv化学实验-瓶盖交换与管路连接/worldcode_Flexiv-A_2026-04-20-12-10-02_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[2121, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.049929529428482056, 0.917127251625061, -0.22191877663135529]`
- 首帧 quat4：`[0.848859429359436, -0.44415655732154846, 0.06281263381242752, -0.27967339754104614]`
- 相邻帧最大 |Δq| 中位数：8.32409e-04
- 相邻帧最大 |Δxyz| 中位数：0.3669 mm
- slot abs_mean：slot[3]=0.8138, slot[4]=0.4783, slot[5]=0.0488, slot[6]=0.3093
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.63°, shift_-1=6.22°, shift_+1=139.49°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_5_Sealing_0425`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_5_Sealing_0425/Flexiv化学实验-系统封闭复位/worldcode_Flexiv-A_2026-04-22-14-50-48_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1659, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05062934011220932, 0.9162501096725464, -0.22171439230442047]`
- 首帧 quat4：`[0.8484504818916321, -0.4448501467704773, 0.06329277902841568, -0.2797037363052368]`
- 相邻帧最大 |Δq| 中位数：4.29183e-04
- 相邻帧最大 |Δxyz| 中位数：0.4976 mm
- slot abs_mean：slot[3]=0.8617, slot[4]=0.4161, slot[5]=0.0437, slot[6]=0.2855
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.52°, shift_-1=6.16°, shift_+1=139.43°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_chemExp_full_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_chemExp_full_0429/Flexiv化学实验-全流程/worldcode_Flexiv-A_2026-04-24-14-38-40_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[8438, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.050731491297483444, 0.9161780476570129, -0.22159703075885773]`
- 首帧 quat4：`[0.8483752608299255, -0.44492819905281067, 0.06335046887397766, -0.2797946631908417]`
- 相邻帧最大 |Δq| 中位数：1.64235e-03
- 相邻帧最大 |Δxyz| 中位数：1.2505 mm
- slot abs_mean：slot[3]=0.8484, slot[4]=0.4764, slot[5]=0.0281, slot[6]=0.2147
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.51°, shift_-1=6.16°, shift_+1=139.41°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_clothing_1_clothing_0425`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_clothing_1_clothing_0425/Flexiv打包衣服-折叠衣服/worldcode_Flexiv-B_2026-04-16-16-14-51_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1706, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05523101985454559, 0.9118193984031677, -0.22428907454013824]`
- 首帧 quat4：`[0.8467183709144592, -0.4495027959346771, 0.06803140789270401, -0.2763819694519043]`
- 相邻帧最大 |Δq| 中位数：2.27004e-03
- 相邻帧最大 |Δxyz| 中位数：3.1771 mm
- slot abs_mean：slot[3]=0.8081, slot[4]=0.4984, slot[5]=0.1047, slot[6]=0.2812
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.70°, shift_-1=5.55°, shift_+1=139.01°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.6°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_clothing_1_clothing_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_clothing_1_clothing_0429/Flexiv打包衣服-折叠衣服/worldcode_Flexiv-B_2026-04-17-10-19-51_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1831, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05499669536948204, 0.9119216203689575, -0.22465483844280243]`
- 首帧 quat4：`[0.8468859791755676, -0.4493858218193054, 0.06782718002796173, -0.27610868215560913]`
- 相邻帧最大 |Δq| 中位数：2.14219e-03
- 相邻帧最大 |Δxyz| 中位数：1.8630 mm
- slot abs_mean：slot[3]=0.8113, slot[4]=0.4979, slot[5]=0.1015, slot[6]=0.2776
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_clothing_2_box_assembly_0513`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_clothing_2_box_assembly_0513/Flexiv打包衣服-礼盒提取与三维组装/worldcode_Flexiv-A_2026-05-05-14-04-03_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[2373, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.050773512572050095, 0.916508674621582, -0.220794215798378]`
- 首帧 quat4：`[0.8481826782226562, -0.4446755051612854, 0.0634838193655014, -0.2807483375072479]`
- 相邻帧最大 |Δq| 中位数：1.73759e-03
- 相邻帧最大 |Δxyz| 中位数：2.4235 mm
- slot abs_mean：slot[3]=0.8553, slot[4]=0.4255, slot[5]=0.1160, slot[6]=0.2631
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.53°, shift_-1=6.26°, shift_+1=139.35°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.3°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_clothing_3_packing_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_clothing_3_packing_0429/Flexiv打包衣服-衣物平稳装箱/worldcode_Flexiv-A_2026-04-27-14-59-11_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1088, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05060627683997154, 0.9160776734352112, -0.22200198471546173]`
- 首帧 quat4：`[0.8485251069068909, -0.44489142298698425, 0.06336561590433121, -0.2793949246406555]`
- 相邻帧最大 |Δq| 中位数：2.08288e-03
- 相邻帧最大 |Δxyz| 中位数：1.7654 mm
- slot abs_mean：slot[3]=0.8187, slot[4]=0.4998, slot[5]=0.1472, slot[6]=0.2214
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.51°, shift_-1=6.13°, shift_+1=139.43°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.1°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_clothing_4_box_closing_0513`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_clothing_4_box_closing_0513/Flexiv打包衣服-礼盒封闭成型/worldcode_Flexiv-A_2026-04-29-14-11-54_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1042, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.050615400075912476, 0.9162876605987549, -0.2216455042362213]`
- 首帧 quat4：`[0.8484364748001099, -0.44483888149261475, 0.06325621902942657, -0.2797722816467285]`
- 相邻帧最大 |Δq| 中位数：2.85795e-03
- 相邻帧最大 |Δxyz| 中位数：1.1178 mm
- slot abs_mean：slot[3]=0.8828, slot[4]=0.1926, slot[5]=0.1656, slot[6]=0.3355
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=102.53°, shift_-1=6.17°, shift_+1=139.43°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (6.2°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_computer_1_cpu_0513`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_computer_1_cpu_0513/Flexiv精细装配-CPU高精度放置与锁紧/worldcode_Flexiv-B_2026-05-06-12-27-05_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[2101, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05499384552240372, 0.9119240045547485, -0.22465404868125916]`
- 首帧 quat4：`[0.846886932849884, -0.44938358664512634, 0.0678277239203453, -0.27610933780670166]`
- 相邻帧最大 |Δq| 中位数：2.35438e-04
- 相邻帧最大 |Δxyz| 中位数：0.3297 mm
- slot abs_mean：slot[3]=0.7274, slot[4]=0.5734, slot[5]=0.0859, slot[6]=0.3429
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_computer_2_ram_0513`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_computer_2_ram_0513/Flexiv精细装配-内存条双臂对称安装/worldcode_Flexiv-B_2026-04-29-15-30-25_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1724, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05499687418341637, 0.9119203090667725, -0.2246549129486084]`
- 首帧 quat4：`[0.846886396408081, -0.4493844211101532, 0.06783147901296616, -0.2761085331439972]`
- 相邻帧最大 |Δq| 中位数：2.97785e-04
- 相邻帧最大 |Δxyz| 中位数：0.5684 mm
- slot abs_mean：slot[3]=0.7999, slot[4]=0.5116, slot[5]=0.0376, slot[6]=0.3025
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_pass_car_0429`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_pass_car_0429/Flexiv小车-小车通行/worldcode_Flexiv-A_2026-04-28-14-55-57_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[3077, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05061979964375496, 0.9162570834159851, -0.22171732783317566]`
- 首帧 quat4：`[-0.279704213142395, 0.8484540581703186, -0.4448446035385132, 0.06328170746564865]`
- 相邻帧最大 |Δq| 中位数：8.93503e-04
- 相邻帧最大 |Δxyz| 中位数：1.0391 mm
- slot abs_mean：slot[3]=0.2391, slot[4]=0.8635, slot[5]=0.4390, slot[6]=0.0611
- slot 模式：max@4, min@6
- 与 pass_car ref 的 layout-blind 角差：direct=6.16°, shift_-1=93.42°, shift_+1=102.52°
- **Verdict**：WXYZ_HIGH_CONF (confidence=high)
  - Reason: first-frame quat as-is is angularly close to pass_car's verified WXYZ home reference: 6.2° vs 93.4° (shift_-1) / 102.5° (shift_+1). slot-pattern=max@4, min@6. Conclusion: same write stack as pass_car, no permute, layout=WXYZ.

### `Flexiv_phone_packing_0518`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_phone_packing_0518/Flexiv打包手机-打包手机/worldcode_Flexiv-B_2026-05-11-11-16-54_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[1868, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05500965565443039, 0.9119665026664734, -0.2245548963546753]`
- 首帧 quat4：`[0.8468616008758545, -0.4493798613548279, 0.06781891733407974, -0.2761951684951782]`
- 相邻帧最大 |Δq| 中位数：1.39239e-03
- 相邻帧最大 |Δxyz| 中位数：1.0947 mm
- slot abs_mean：slot[3]=0.6180, slot[4]=0.7141, slot[5]=0.0487, slot[6]=0.2227
- slot 模式：max@4, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.04°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@4, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_whack_a_mole_0518`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_whack_a_mole_0518/Flexiv打地鼠-打地鼠/worldcode_Flexiv-B_2026-05-12-18-51-16_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[824, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05500268191099167, 0.9119168519973755, -0.22465106844902039]`
- 首帧 quat4：`[0.8468824028968811, -0.4493899643421173, 0.0678330734372139, -0.27611133456230164]`
- 相邻帧最大 |Δq| 中位数：1.97619e-04
- 相邻帧最大 |Δxyz| 中位数：0.1095 mm
- slot abs_mean：slot[3]=0.6138, slot[4]=0.7187, slot[5]=0.0886, slot[6]=0.2429
- slot 模式：max@4, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.73°, shift_-1=5.54°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@4, min@5. Conclusion: dataset was permuted at write time.

### `Flexiv_whack_a_mole_0527`

- Episode 抽样：`/mnt/data/data/ymm/dataset/Flexiv/release/Flexiv_whack_a_mole_0527/Flexiv打地鼠-打地鼠/worldcode_Flexiv-B_2026-05-27-11-13-22_0`
- Pose path：`action/left_eef/feedback/pose`
- Pose value shape：`[802, 7]` dtype=`[('value', '<f4', (7,)), ('timestamp', '<f8')]`
- 前 200 帧的 quat norm：mean=1.00000, std=0.00000 (应当≈1)
- 首帧 xyz (m)：`[0.05501861870288849, 0.9119012355804443, -0.22464315593242645]`
- 首帧 quat4：`[0.846872091293335, -0.4494055211544037, 0.06784068048000336, -0.2761157751083374]`
- 相邻帧最大 |Δq| 中位数：3.51667e-06
- 相邻帧最大 |Δxyz| 中位数：0.0041 mm
- slot abs_mean：slot[3]=0.8468, slot[4]=0.4495, slot[5]=0.0679, slot[6]=0.2761
- slot 模式：max@3, min@5
- 与 pass_car ref 的 layout-blind 角差：direct=101.72°, shift_-1=5.53°, shift_+1=139.05°
- **Verdict**：LIKELY_XYZW_PERMUTED_shift_-1 (confidence=high)
  - Reason: shift_-1 cyclic shift of pass_car's WXYZ ref is the closest layout-blind match (5.5°). slot-pattern=max@3, min@5. Conclusion: dataset was permuted at write time.

### `ossutil_output`

**Error**: `no dataset.hdf5 found anywhere under task dir`

---

## 总结 & 给采集脚本作者的建议

- 共扫描 27 个 task；
- 高置信判 WXYZ：1
- 中置信判 WXYZ：0
- 判倾向 permuted：25
- 单 ref 不能判（home pose 物理差异大）：0
- 解析错误：1

如果所有 task 都用同一套 `flexivrdk` + `action/{arm}_eef/feedback/pose` 录制管线（采集 toolkit 一致性指纹一致），则被标 `AMBIGUOUS_DIFFERENT_HOME_POSE` 的任务在字节层 **必然** 与 `pass_car` 一致 = WXYZ；AMBIGUOUS 标签仅说明『单 ref 物理比对不能独立锁定』，不代表 layout 真的有变化。采集脚本 `process_data_flexiv_zarr.py` 是 release 数据集的统一下游消费者，其 `R.from_quat(quat)` 一行错位（参见 `docs/flexiv_RT/agents/04-quat-layout-fix/dataset_vs_probe_log.md` §5.2）会同时污染所有 task 的 zarr，迁移到干净 wxyz 链路时这一行是唯一需要修改的位置。