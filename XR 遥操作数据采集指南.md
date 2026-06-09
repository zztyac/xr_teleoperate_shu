# XR 遥操作数据采集指南

本文档按当前项目的 `teleop/teleop_hand_and_arm.py` 采集脚本整理，重点说明使用键盘控制机械臂末端进行车内物理按键数据采集的流程。键盘采集模式下，PC 键盘负责控制末端位置、姿态和 BrainCo 食指按压状态，VR 端主要用于第一视角观察和图像回传。

## 1. 设备与网络准备

所有设备需要在同一个局域网和同一网段下，例如 `192.168.123.xxx`。

- PC 端使用网线连接路由器。
- 机器人端使用网线连接路由器。
- VR 头显连接同一路由器的 5G Wi-Fi。
- 机器人上电后确认关节受控，周围留出安全空间。

常用机器人 IP：

```bash
192.168.123.164
```

### 1.1 副驾驶位远程采集方式

采集员可以拿自己的电脑进入副驾驶位完成全流程采集。自己的电脑只作为远程终端和键盘输入入口，实际采集脚本仍然运行在项目 PC 上。

操作方式：

1. 自己的电脑连接到同一个局域网，例如同一路由器的 5G Wi-Fi。
2. 确认项目 PC 的局域网 IP，例如 `192.168.123.xxx`。
3. 在自己的电脑终端 SSH 登录项目 PC：

```bash
ssh ubuntu@192.168.123.xxx
```

4. 登录后，在这个 SSH 终端里执行后续 PC 端启动指令，例如 `conda activate tv`、`cd /home/ubuntu/zty/xr_teleoperate_shu`、图像测试和采集脚本启动命令。
5. VR 头显访问 WebXR 页面时，也使用项目 PC 的 IP：

```text
https://192.168.123.xxx:8012?ws=wss://192.168.123.xxx:8012
```

键盘采集时，`r`、`Enter`、`1`-`5`、`w/s/a/d/e/c`、`f` 等按键需要发送到正在运行采集脚本的 SSH 终端。因此采集过程中保持自己的电脑终端焦点在该 SSH 会话里。

## 2. 机器人端启动服务

通过 PC SSH 登录机器人：

```bash
ssh unitree@192.168.123.164
```

默认密码通常为：

```bash
123
```

### 2.1 启动 BrainCo 灵巧手服务

在机器人端终端执行：

```bash
sudo ~/brainco_hand_service/bin/brainco_hand_server
```

保持该终端开启。

### 2.2 启动图像服务

另开一个机器人端终端：

```bash
conda activate teleimager
python -m teleimager.image_server --rs
```

保持该终端开启。当前项目的 `teleimager` 子模块已接入多相机配置，采集脚本会按相机配置读取头部、腕部和可选右侧相机图像。

## 3. PC 端启动采集程序

以下命令都在项目 PC 上执行。如果人在副驾驶位使用自己的电脑采集，先按 1.1 小节 SSH 到项目 PC，再在 SSH 终端里执行这些命令。

在 PC 端进入项目目录：

```bash
conda activate tv
cd /home/ubuntu/zty/xr_teleoperate_shu
```

### 3.1 测试图像链路

正式采集前先确认图像能正常接收：

```bash
python -m teleimager.image_client --host 192.168.123.164
```

确认画面清晰、不卡顿、相机视角覆盖操作区域后，按 `Ctrl+C` 结束测试。

### 3.2 推荐键盘采集启动命令

车内物理按键任务推荐使用键盘控制末端。建议加 `--headless`，它只关闭 Rerun 可视化，不会关闭终端日志，也能减少按 Enter 开始录制时的主循环阻塞。

```bash
python teleop/teleop_hand_and_arm.py \
  --record \
  --task-name vehicle_physical_button_press \
  --headless
```

如果需要使用 `vehicle_physical_button_press_ccw` 这套初始姿态：

```bash
python teleop/teleop_hand_and_arm.py \
  --record \
  --task-name vehicle_physical_button_press_ccw \
  --headless
```

`vehicle_physical_button_press` 和 `vehicle_physical_button_press_ccw` 都会自动启用 1-5 子任务标签写入 `data.json -> text`。如果使用其他 task name，需要扩展脚本中的子任务 task name 集合，或通过 `--task-goal`、`--task-desc`、`--task-steps` 手动指定。

### 3.3 运控模式和调试模式

运控模式：

```bash
python teleop/teleop_hand_and_arm.py \
  --record \
  --motion \
  --task-name vehicle_physical_button_press \
  --headless
```

Debug 模式：

```bash
python teleop/teleop_hand_and_arm.py \
  --record \
  --task-name vehicle_physical_button_press \
  --headless
```

Debug 模式下，当前 G1_29 配置会优先从 `teleop/initial_target_poses.json` 读取 `all_joint_q`，并控制腿、腰、手臂进入配置姿态。如果只希望控制双臂，不控制腿和腰：

```bash
python teleop/teleop_hand_and_arm.py \
  --record \
  --task-name vehicle_physical_button_press \
  --debug-arms-only \
  --headless
```

## 4. 初始姿态配置

采集脚本启动时会固定从以下文件读取初始姿态：

```bash
teleop/initial_target_poses.json
```

读取规则：

- `--task-name` 对应 JSON 顶层 key。
- `--arm` 默认是 `G1_29`，对应 JSON 内的机器人型号 key。
- `dual_arm_q` 用作双臂 IK 和控制初始姿态。
- `all_joint_q` 用作 G1_29 Debug 全身初始化姿态。

如果启动时报初始姿态缺失，需要检查 `--task-name` 是否和 JSON key 一致。

## 5. VR 端观察画面

在 VR 头显浏览器打开 PC 端 WebXR 地址，将 `192.168.123.xxx` 替换为 PC 的局域网 IP：

```text
https://192.168.123.xxx:8012?ws=wss://192.168.123.xxx:8012
```

进入网页后点击 `Enter Virtual Reality`。键盘采集时，VR 端主要用于观察机器人视角；机械臂末端动作由 PC 键盘控制。

推荐顺序：

1. 机器人端启动 BrainCo 手服务和图像服务。
2. PC 端启动采集脚本。
3. VR 端进入 WebXR 页面并确认画面。
4. PC 端采集脚本终端按 `r` 开始跟随和控制。

## 6. 键盘采集按键说明

### 6.1 程序状态按键

| 按键 | 功能 |
| --- | --- |
| `r` | 开始机器人跟随键盘/XR 输入 |
| `q` | 停止程序并退出，退出时手臂会执行回初始姿态流程 |
| `Enter` | 开始当前 episode 录制，或保存正在录制的 episode |

不要再用 `s` 开始/结束录制。当前脚本里 `s` 是末端移动按键，录制切换统一使用 `Enter`。

### 6.2 子任务选择按键

子任务只能在未录制时切换。录制中不能切换，需要先按 `Enter` 保存当前 episode。

| 按键 | 功能 |
| --- | --- |
| `1`-`5` | 直接选择对应按键子任务 |
| `n` | 切换到下一个子任务 |
| `p` | 切换到上一个子任务 |

当前内置子任务：

| 编号 | 子任务 ID | 训练语言标签 |
| --- | --- | --- |
| `1` | `front_windshield_defrost` | `Press the front windshield defrost button once.` |
| `2` | `ac_temperature_down` | `Press the air conditioning temperature down button once.` |
| `3` | `fan_speed_down` | `Press the fan speed down button once.` |
| `4` | `trunk_open_long_press` | `Long-press the trunk open button until the trunk starts opening.` |
| `5` | `sunshade_open_long_press` | `Long-press the sunshade open button until the sunshade starts opening.` |

### 6.3 末端平移按键

按住按键持续移动，松开停止。方向是机器人坐标系方向。

| 按键 | 方向 |
| --- | --- |
| `w` | 末端沿机器人 X 轴正方向移动 |
| `s` | 末端沿机器人 X 轴负方向移动 |
| `a` | 末端沿机器人 Y 轴正方向移动 |
| `d` | 末端沿机器人 Y 轴负方向移动 |
| `e` | 末端沿机器人 Z 轴正方向移动 |
| `c` | 末端沿机器人 Z 轴负方向移动 |

默认线速度：

```bash
--keyboard-linear-speed 0.025
```

### 6.4 末端旋转按键

按住按键持续旋转，松开停止。

| 按键 | 方向 |
| --- | --- |
| `u` | 绕机器人 X 轴负方向旋转 |
| `o` | 绕机器人 X 轴正方向旋转 |
| `i` | 绕机器人 Y 轴正方向旋转 |
| `k` | 绕机器人 Y 轴负方向旋转 |
| `j` | 绕机器人 Z 轴正方向旋转 |
| `l` | 绕机器人 Z 轴负方向旋转 |

默认角速度：

```bash
--keyboard-angular-speed 0.35
```

### 6.5 手/夹爪动作按键

| 按键 | 功能 |
| --- | --- |
| `f` | 切换当前键盘控制手臂的食指/夹爪按压状态 |
| `space` | 当前键盘控制末端平滑回到初始姿态，并关闭按压状态 |

默认只控制右臂：

```bash
--keyboard-arm right
```

也可以改为左臂或双臂：

```bash
--keyboard-arm left
--keyboard-arm both
```

## 7. 推荐键盘采集流程

每个 episode 建议按以下顺序执行：

1. 启动脚本后，确认终端提示 `Press [r] to start syncing...`。
2. 在 VR 或显示器中确认相机画面正常。
3. 在 PC 终端按 `r`，机器人开始接收键盘控制目标。
4. 录制前按 `1`-`5` 选择本次子任务。
5. 使用 `w/s/a/d/e/c` 和 `u/o/i/k/j/l` 将右手食指对准目标按键。
6. 按 `Enter` 开始录制当前 episode。
7. 使用键盘完成演示：
   - 短按任务：移动到按钮表面，按 `f` 开启按压，按钮触发后再按 `f` 关闭按压并撤回。
   - 长按任务：按 `f` 开启按压并保持到目标功能开始，再按 `f` 关闭按压。
8. 按 `Enter` 保存当前 episode。
9. 需要回初始姿态时按 `space`，等待终端提示回到初始姿态。
10. 选择下一个子任务，重复采集。

示例采集顺序：

```text
r
1
Enter  开始录制前窗除雾
键盘完成按压
Enter  保存
2
Enter  开始录制空调温度降低
键盘完成按压
Enter  保存
...
```

## 8. 数据保存与终端日志

默认数据目录：

```bash
/mnt/data/zty/json_data/<task-name>/episode_xxxx/data.json
```

例如：

```bash
/mnt/data/zty/json_data/vehicle_physical_button_press/episode_0001/data.json
```

可以用 `--task-dir` 改根目录：

```bash
python teleop/teleop_hand_and_arm.py \
  --record \
  --task-dir /mnt/data/zty/json_data/ \
  --task-name vehicle_physical_button_press \
  --headless
```

`--headless` 后终端仍会打印逐帧采集进度：

```text
==> episode_id:0  item_id:12  current_time:...
```

保存 episode 时会关闭 `data.json` 中的 `data` 数组，并把图像文件写入 episode 目录下的 `colors/`、`depths/` 等子目录。

## 9. 常见问题

### 9.1 按 Enter 时手臂掉一下

开始录制时如果启用 Rerun，可视化初始化可能阻塞主控制循环一小段时间。键盘采集建议加：

```bash
--headless
```

它不会关闭终端日志，只关闭 Rerun 可视化。

### 9.2 加了 `--headless` 后没有 Rerun 窗口

这是正常现象。`--headless` 的作用就是不启动 Rerun。终端仍会打印 episode 和 item 进度。

### 9.3 子任务标签没有写入 `data.json -> text`

当前脚本会在以下 task name 自动启用 1-5 子任务标签写入：

```bash
--task-name vehicle_physical_button_press
--task-name vehicle_physical_button_press_ccw
```

如果使用其他 task name，需要扩展脚本中的子任务 task name 集合，或通过 `--task-goal`、`--task-desc`、`--task-steps` 手动指定。

### 9.4 按键没有反应

先检查：

- 终端是否已经按过 `r`。
- 焦点是否在运行采集脚本的终端窗口。
- 是否用了 `--ipc`。当前键盘末端控制只在 `controller` 输入模式且非 IPC 键盘监听下启用。
- 是否正在录制中切换子任务。录制中 `1`-`5`、`n`、`p` 不允许切换子任务。

### 9.5 只想调慢或调快键盘控制速度

调线速度：

```bash
--keyboard-linear-speed 0.015
```

调角速度：

```bash
--keyboard-angular-speed 0.20
```

## 10. 代码仓库与子模块

当前主项目：

```text
git@github.com:zztyac/xr_teleoperate_shu.git
```

当前项目已将相关子模块指向自有 fork：

```text
git@github.com:zztyac/teleimager_shu.git
git@github.com:zztyac/televuer_shu.git
```

拉取项目时建议使用：

```bash
git clone --recurse-submodules git@github.com:zztyac/xr_teleoperate_shu.git
```

如果已经 clone 过主仓库：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```
