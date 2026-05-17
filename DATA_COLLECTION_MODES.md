# Data Collection Modes and Recorded Data Format

本文档说明当前 `teleop/teleop_hand_and_arm.py` 在不同运行模式下能够采集并写入的数据格式。

代码依据：

- 录制入口：`teleop/teleop_hand_and_arm.py`
- 数据写盘：`teleop/utils/episode_writer.py`
- XR 输入结构：`teleop/televuer/src/televuer/tv_wrapper.py`
- 灵巧手控制：`teleop/robot_control/robot_hand_*.py`

## 1. 当前默认采集配置

当前主程序默认参数如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--frequency` | `15.0` | 主循环控制频率和录制元数据中的 FPS |
| `--input-mode` | `hand` | 使用 XR 手势追踪 |
| `--ee` | `brainco` | 使用 BrainCo 灵巧手 |
| `--arm` | `G1_29` | 默认机械臂型号 |
| `--record` | 默认关闭 | 需要显式加 `--record` 才写数据 |
| `--task-name` | `vehicle_physical_button_press` | 默认任务目录名 |

默认采集命令示例：

```bash
python teleop_hand_and_arm.py --record
```

该命令等价于：

```bash
python teleop_hand_and_arm.py --frequency 15 --input-mode hand --ee brainco --arm G1_29 --record
```

注意：`--frequency` 控制主循环和录制元数据中的 FPS。相机服务自身的采集 FPS 来自 `teleop/teleimager/cam_config_server.yaml`，当前没有在主入口中强制覆盖。

## 2. 录制文件结构

录制开启后，数据默认保存到：

```text
./utils/data/<task-name>/episode_0001/
```

目录结构示例：

```text
episode_0001/
  data.json
  colors/
    000000_color_0.jpg
    000000_color_1.jpg
    000000_color_2.jpg
    000000_color_3.jpg
  depths/
  audios/
```

`data.json` 顶层结构：

```json
{
  "info": {
    "version": "1.0.0",
    "date": "2026-05-18",
    "author": "unitree",
    "image": {
      "width": 640,
      "height": 480,
      "fps": 15.0
    },
    "depth": {
      "width": 640,
      "height": 480,
      "fps": 15.0
    },
    "audio": {
      "sample_rate": 16000,
      "channels": 1,
      "format": "PCM",
      "bits": 16
    },
    "joint_names": {
      "left_arm": [],
      "left_ee": [],
      "right_arm": [],
      "right_ee": [],
      "body": []
    },
    "tactile_names": {
      "left_ee": [],
      "right_ee": []
    },
    "sim_state": ""
  },
  "text": {
    "goal": "Press in-car physical buttons with the BrainCo dexterous hand.",
    "desc": "Collect 15 FPS demonstrations for in-car physical button press testing.",
    "steps": "step1: move the BrainCo dexterous hand to the target button; step2: align the fingertip with the button surface; step3: press the button; step4: release and return to a safe pose;"
  },
  "data": []
}
```

每一帧 item 的固定结构：

```json
{
  "idx": 0,
  "colors": {},
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

字段含义：

| 字段 | 含义 | 当前写入情况 |
| --- | --- | --- |
| `idx` | 帧序号，从 0 开始 | 始终写入 |
| `colors` | 彩色图像路径 | 有启用 ZMQ 的相机且拿到图像时写入 |
| `depths` | 深度图路径 | 当前主流程为空对象 `{}` |
| `states.*.qpos` | 当前状态 | 按模式写入 |
| `states.*.qvel` | 当前速度 | 当前为空数组 |
| `states.*.torque` | 当前力矩 | 当前为空数组 |
| `actions.*.qpos` | 控制目标或动作 | 按模式写入 |
| `actions.*.qvel` | 速度动作 | 当前为空数组 |
| `actions.*.torque` | 力矩动作 | 当前为空数组 |
| `tactiles` | 触觉数据 | 当前为 `null` |
| `audios` | 音频数据 | 当前为 `null` |
| `sim_state` | 仿真状态 | 仅 `--sim` 时写入，否则为 `null` |

## 3. 图像数据格式

图像是否写入取决于相机配置中的 `enable_zmq` 和主循环是否成功拿到图像。

当前保存后，`colors` 中存的是相对路径，不是图像矩阵。

### 3.1 头部双目相机模式

当 `camera_config["head_camera"]["binocular"] == true`：

| key | 内容 |
| --- | --- |
| `color_0` | 头部双目图像左半部分 |
| `color_1` | 头部双目图像右半部分 |
| `color_2` | 左腕相机 |
| `color_3` | 右腕相机 |

示例：

```json
{
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg",
    "color_2": "colors/000000_color_2.jpg",
    "color_3": "colors/000000_color_3.jpg"
  }
}
```

### 3.2 头部单目相机模式

当 `camera_config["head_camera"]["binocular"] == false`：

| key | 内容 |
| --- | --- |
| `color_0` | 头部相机 |
| `color_1` | 左腕相机 |
| `color_2` | 右腕相机 |

示例：

```json
{
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg",
    "color_2": "colors/000000_color_2.jpg"
  }
}
```

如果某一路相机未启用 ZMQ，或该帧没有拿到图像，对应 key 不会出现在 `colors` 中。

## 4. 机械臂数据格式

所有录制模式都会写入双臂状态和动作：

```json
{
  "states": {
    "left_arm": {
      "qpos": [0.11, -0.22, 0.03, 0.44, -0.15, 0.06, 0.17],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.12, 0.21, -0.04, 0.43, 0.14, -0.05, 0.18],
      "qvel": [],
      "torque": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.12, -0.20, 0.05, 0.46, -0.12, 0.08, 0.20],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.10, 0.19, -0.02, 0.45, 0.12, -0.03, 0.21],
      "qvel": [],
      "torque": []
    }
  }
}
```

字段来源：

| 字段 | 来源 |
| --- | --- |
| `states.left_arm.qpos` | 当前左臂关节位置 |
| `states.right_arm.qpos` | 当前右臂关节位置 |
| `actions.left_arm.qpos` | IK 求解出的左臂目标关节位置 |
| `actions.right_arm.qpos` | IK 求解出的右臂目标关节位置 |

当前主流程按每侧 7 维切分：

```python
left_arm_state = current_lr_arm_q[:7]
right_arm_state = current_lr_arm_q[-7:]
left_arm_action = sol_q[:7]
right_arm_action = sol_q[-7:]
```

对于 `G1_29`、`H1_2`、`H2` 这类 7 DoF 单臂配置，顺序通常是：

```text
left_arm/right_arm:
  [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
```

对于 `G1_23`、`H1` 等实际手臂自由度较少的配置，代码仍按左右切片写入，具体长度以对应控制器返回的数组为准。

## 5. `--input-mode hand --ee brainco`

这是当前默认模式，适合 BrainCo 灵巧手做车内物理按键按压采集。

启动示例：

```bash
python teleop_hand_and_arm.py --record
```

或显式写全：

```bash
python teleop_hand_and_arm.py --input-mode hand --ee brainco --frequency 15 --record
```

### 5.1 可采集数据

| 数据块 | 是否完整写入 | 维度 |
| --- | --- | --- |
| 图像 `colors` | 是，取决于相机配置 | 1 到 4 路 jpg |
| 左臂状态 `states.left_arm.qpos` | 是 | 通常 7 |
| 右臂状态 `states.right_arm.qpos` | 是 | 通常 7 |
| 左臂动作 `actions.left_arm.qpos` | 是 | 通常 7 |
| 右臂动作 `actions.right_arm.qpos` | 是 | 通常 7 |
| 左手状态 `states.left_ee.qpos` | 是 | 6 |
| 右手状态 `states.right_ee.qpos` | 是 | 6 |
| 左手动作 `actions.left_ee.qpos` | 是 | 6 |
| 右手动作 `actions.right_ee.qpos` | 是 | 6 |
| body 状态/动作 | 空 | `[]` |
| sim_state | 仅 `--sim` | 取决于仿真订阅数据 |

### 5.2 BrainCo 手部 qpos 顺序

每只手 6 维：

```text
[thumb, thumb_aux, index, middle, ring, pinky]
```

`states.left_ee.qpos` 和 `states.right_ee.qpos` 是硬件反馈状态。

`actions.left_ee.qpos` 和 `actions.right_ee.qpos` 是由 XR 手部 25 点重定向得到的目标动作，当前 BrainCo 控制器会归一化到接近 `[0, 1]` 的范围：

```text
0.0 = fully open
1.0 = fully closed
```

### 5.3 单帧示例

```json
{
  "idx": 0,
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg",
    "color_2": "colors/000000_color_2.jpg",
    "color_3": "colors/000000_color_3.jpg"
  },
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [0.10, -0.20, 0.03, 0.40, -0.10, 0.05, 0.12],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.08, 0.18, -0.02, 0.38, 0.09, -0.04, 0.11],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.02, 0.04, 0.10, 0.12, 0.08, 0.05],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.15, 0.18, 0.72, 0.35, 0.20, 0.16],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.11, -0.19, 0.04, 0.42, -0.09, 0.06, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.07, 0.17, -0.01, 0.40, 0.08, -0.03, 0.12],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.03, 0.05, 0.12, 0.14, 0.09, 0.06],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.18, 0.20, 0.85, 0.42, 0.25, 0.19],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

### 5.4 `--input-mode controller --ee brainco`

该模式使用 controller 的 B/Y 类按键控制 BrainCo 固定姿态，不依赖手势 25 点数据。

启动示例：

```bash
python teleop_hand_and_arm.py --input-mode controller --ee brainco --record
```

按键映射：

| 按键 | 效果 |
| --- | --- |
| `left_ctrl_bButton` | 长按时左手只伸出食指，其余手指闭合；松开后左手全部张开 |
| `right_ctrl_bButton` | 长按时右手只伸出食指，其余手指闭合；松开后右手全部张开 |

BrainCo qpos 顺序仍是：

```text
[thumb, thumb_aux, index, middle, ring, pinky]
```

全部张开：

```text
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

只伸出食指：

```text
[1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
```

该模式会完整记录 BrainCo 双手状态/动作，并且像其他 controller 模式一样记录 body 状态和 3 维 body 动作。

单帧核心字段示例：

```json
{
  "states": {
    "left_ee": {
      "qpos": [0.02, 0.04, 0.10, 0.12, 0.08, 0.05],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.03, 0.05, 0.11, 0.10, 0.07, 0.04],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": [0.01, -0.02, 0.00, 0.08, -0.05, 0.03]
    }
  },
  "actions": {
    "left_ee": {
      "qpos": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": [0.12, -0.03, -0.06]
    }
  }
}
```

## 6. `--input-mode hand --ee inspire_dfx`

启动示例：

```bash
python teleop_hand_and_arm.py --input-mode hand --ee inspire_dfx --record
```

### 6.1 可采集数据

| 数据块 | 是否完整写入 | 维度 |
| --- | --- | --- |
| 图像 `colors` | 是，取决于相机配置 | 1 到 4 路 jpg |
| 双臂状态/动作 | 是 | 通常每侧 7 |
| 左手状态/动作 | 是 | 6 |
| 右手状态/动作 | 是 | 6 |
| body 状态/动作 | 空 | `[]` |

### 6.2 Inspire qpos 顺序

每只手 6 维：

```text
[pinky, ring, middle, index, thumb_bend, thumb_rotation]
```

### 6.3 单帧示例

```json
{
  "idx": 0,
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg",
    "color_2": "colors/000000_color_2.jpg",
    "color_3": "colors/000000_color_3.jpg"
  },
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [0.10, -0.20, 0.03, 0.40, -0.10, 0.05, 0.12],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.08, 0.18, -0.02, 0.38, 0.09, -0.04, 0.11],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [1.0, 1.0, 1.0, 0.95, 0.90, 0.88],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.92, 0.90, 0.86, 0.35, 0.42, 0.40],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.11, -0.19, 0.04, 0.42, -0.09, 0.06, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.07, 0.17, -0.01, 0.40, 0.08, -0.03, 0.12],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.98, 0.98, 0.96, 0.90, 0.88, 0.86],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.90, 0.88, 0.82, 0.20, 0.38, 0.36],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

## 7. `--input-mode hand --ee inspire_ftp`

启动示例：

```bash
python teleop_hand_and_arm.py --input-mode hand --ee inspire_ftp --record
```

### 7.1 可采集数据

`inspire_ftp` 的录制结构与 `inspire_dfx` 相同：

| 数据块 | 是否完整写入 | 维度 |
| --- | --- | --- |
| 图像 `colors` | 是，取决于相机配置 | 1 到 4 路 jpg |
| 双臂状态/动作 | 是 | 通常每侧 7 |
| 左手状态/动作 | 是 | 6 |
| 右手状态/动作 | 是 | 6 |
| body 状态/动作 | 空 | `[]` |

每只手 qpos 顺序：

```text
[pinky, ring, middle, index, thumb_bend, thumb_rotation]
```

### 7.2 单帧示例

```json
{
  "idx": 0,
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg"
  },
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [0.10, -0.20, 0.03, 0.40, -0.10, 0.05, 0.12],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.08, 0.18, -0.02, 0.38, 0.09, -0.04, 0.11],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [900, 900, 880, 820, 760, 740],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [850, 830, 780, 250, 420, 410],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.11, -0.19, 0.04, 0.42, -0.09, 0.06, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.07, 0.17, -0.01, 0.40, 0.08, -0.03, 0.12],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.90, 0.90, 0.88, 0.82, 0.76, 0.74],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.85, 0.83, 0.78, 0.25, 0.42, 0.41],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

注意：FTP 控制器内部会把动作值缩放成硬件命令，但录制到 `actions.*.qpos` 的仍是归一化后的 `left_q_target/right_q_target`。状态值来自硬件反馈，具体量纲取决于 SDK 返回的 `angle_act`。

## 8. `--input-mode hand --ee dex3`

启动示例：

```bash
python teleop_hand_and_arm.py --input-mode hand --ee dex3 --record
```

### 8.1 可采集数据

| 数据块 | 是否完整写入 | 维度 |
| --- | --- | --- |
| 图像 `colors` | 是，取决于相机配置 | 1 到 4 路 jpg |
| 双臂状态/动作 | 是 | 通常每侧 7 |
| 左手状态/动作 | 是 | 7 |
| 右手状态/动作 | 是 | 7 |
| body 状态/动作 | 空 | `[]` |

### 8.2 Dex3 qpos 顺序

左手：

```text
[left_thumb_0, left_thumb_1, left_thumb_2, left_middle_0, left_middle_1, left_index_0, left_index_1]
```

右手：

```text
[right_thumb_0, right_thumb_1, right_thumb_2, right_index_0, right_index_1, right_middle_0, right_middle_1]
```

### 8.3 单帧示例

```json
{
  "idx": 0,
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg",
    "color_2": "colors/000000_color_2.jpg",
    "color_3": "colors/000000_color_3.jpg"
  },
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [0.10, -0.20, 0.03, 0.40, -0.10, 0.05, 0.12],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.08, 0.18, -0.02, 0.38, 0.09, -0.04, 0.11],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.02, 0.03, 0.04, 0.08, 0.09, 0.10, 0.11],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.03, 0.04, 0.05, 0.62, 0.68, 0.18, 0.20],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.11, -0.19, 0.04, 0.42, -0.09, 0.06, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.07, 0.17, -0.01, 0.40, 0.08, -0.03, 0.12],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.02, 0.04, 0.05, 0.10, 0.11, 0.12, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.04, 0.05, 0.06, 0.78, 0.82, 0.22, 0.25],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

## 9. `--input-mode hand --ee dex1`

该模式使用手势 pinch 控制左右夹爪。

启动示例：

```bash
python teleop_hand_and_arm.py --input-mode hand --ee dex1 --record
```

### 9.1 可采集数据

| 数据块 | 是否完整写入 | 维度 |
| --- | --- | --- |
| 图像 `colors` | 是，取决于相机配置 | 1 到 4 路 jpg |
| 双臂状态/动作 | 是 | 通常每侧 7 |
| 左夹爪状态/动作 | 是 | 1 |
| 右夹爪状态/动作 | 是 | 1 |
| body 状态/动作 | 空 | `[]` |

### 9.2 Dex1 qpos 含义

每只夹爪 1 维：

```text
[gripper]
```

输入来源：

```text
left_gripper_value = tele_data.left_hand_pinchValue
right_gripper_value = tele_data.right_hand_pinchValue
```

### 9.3 单帧示例

```json
{
  "idx": 0,
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg"
  },
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [0.10, -0.20, 0.03, 0.40, -0.10, 0.05, 0.12],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.08, 0.18, -0.02, 0.38, 0.09, -0.04, 0.11],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.20],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.85],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.11, -0.19, 0.04, 0.42, -0.09, 0.06, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.07, 0.17, -0.01, 0.40, 0.08, -0.03, 0.12],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.22],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.90],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

## 10. `--input-mode controller --ee dex1`

该模式使用手柄扳机控制夹爪，并且在 `--motion` 开启时可以记录 body 移动动作。

启动示例：

```bash
python teleop_hand_and_arm.py --input-mode controller --ee dex1 --record
```

如果需要底盘/身体运动控制：

```bash
python teleop_hand_and_arm.py --input-mode controller --ee dex1 --motion --record
```

### 10.1 可采集数据

| 数据块 | 是否完整写入 | 维度 |
| --- | --- | --- |
| 图像 `colors` | 是，取决于相机配置 | 1 到 4 路 jpg |
| 双臂状态/动作 | 是 | 通常每侧 7 |
| 左夹爪状态/动作 | 是 | 1 |
| 右夹爪状态/动作 | 是 | 1 |
| body 状态 | 是 | 取决于 `arm_ctrl.get_current_motor_q()` |
| body 动作 | 是 | 3 |

### 10.2 Dex1 qpos 含义

每只夹爪 1 维：

```text
[gripper]
```

输入来源：

```text
left_gripper_value = tele_data.left_ctrl_triggerValue
right_gripper_value = tele_data.right_ctrl_triggerValue
```

body 动作顺序：

```text
[
  -left_ctrl_thumbstickValue[1] * 0.3,
  -left_ctrl_thumbstickValue[0] * 0.3,
  -right_ctrl_thumbstickValue[0] * 0.3
]
```

可理解为：

```text
[forward_backward_velocity, lateral_velocity, yaw_velocity]
```

### 10.3 单帧示例

```json
{
  "idx": 0,
  "colors": {
    "color_0": "colors/000000_color_0.jpg",
    "color_1": "colors/000000_color_1.jpg",
    "color_2": "colors/000000_color_2.jpg",
    "color_3": "colors/000000_color_3.jpg"
  },
  "depths": {},
  "states": {
    "left_arm": {
      "qpos": [0.10, -0.20, 0.03, 0.40, -0.10, 0.05, 0.12],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.08, 0.18, -0.02, 0.38, 0.09, -0.04, 0.11],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.35],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.80],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": [0.01, -0.02, 0.00, 0.08, -0.05, 0.03, 0.10, 0.02, 0.01, -0.01, 0.04, -0.03, 0.12, 0.08, -0.06, 0.03, 0.02, 0.01, -0.02, 0.05, 0.06, -0.04, 0.01, 0.03, -0.02, 0.07, 0.00, 0.02, -0.01]
    }
  },
  "actions": {
    "left_arm": {
      "qpos": [0.11, -0.19, 0.04, 0.42, -0.09, 0.06, 0.13],
      "qvel": [],
      "torque": []
    },
    "right_arm": {
      "qpos": [-0.07, 0.17, -0.01, 0.40, 0.08, -0.03, 0.12],
      "qvel": [],
      "torque": []
    },
    "left_ee": {
      "qpos": [0.40],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [0.85],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": [0.12, -0.03, -0.06]
    }
  },
  "tactiles": null,
  "audios": null,
  "sim_state": null
}
```

注意：`states.body.qpos` 的长度由所选 `--arm` 对应控制器的全身电机状态决定。`G1_29` 通常是 29 维。

## 11. `--sim` 仿真模式

任何上述录制组合都可以附加 `--sim`。

示例：

```bash
python teleop_hand_and_arm.py --input-mode hand --ee brainco --sim --record
```

开启 `--sim` 后，每帧额外写入 `sim_state`：

```json
{
  "idx": 0,
  "sim_state": {
    "example_key": "example_value"
  }
}
```

`sim_state` 的具体结构来自 `teleop/utils/sim_state_topic.py` 中的订阅数据，不由 `EpisodeWriter` 固定定义。

## 12. 不完整或不推荐的组合

当前代码只在以下组合中给末端执行器写入完整状态/动作：

```text
hand + brainco
controller + brainco
hand + inspire_dfx
hand + inspire_ftp
hand + dex3
hand + dex1
controller + dex1
```

下面这些组合能运行到某些公共流程，但末端执行器数据不完整：

| 组合 | 结果 |
| --- | --- |
| `controller + dex3` | `left_ee/right_ee` 为空数组 |
| `controller + inspire_dfx` | `left_ee/right_ee` 为空数组 |
| `controller + inspire_ftp` | `left_ee/right_ee` 为空数组 |

这类组合仍会写入图像和双臂数据，但不适合做完整末端执行器 imitation learning 数据集。

示例：

```json
{
  "states": {
    "left_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  },
  "actions": {
    "left_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "right_ee": {
      "qpos": [],
      "qvel": [],
      "torque": []
    },
    "body": {
      "qpos": []
    }
  }
}
```

## 13. XR 原始数据与录制数据的区别

XR 层 `TeleData` 实际可以提供更多数据，但当前录制逻辑没有直接写入这些原始字段。

### 13.1 hand tracking 下 XR 能拿到

| 字段 | 形状/类型 | 当前是否直接写入 JSON |
| --- | --- | --- |
| `head_pose` | `(4, 4)` | 否 |
| `left_wrist_pose` | `(4, 4)` | 否，参与 IK |
| `right_wrist_pose` | `(4, 4)` | 否，参与 IK |
| `left_hand_pos` | `(25, 3)` | 否，参与手部重定向 |
| `right_hand_pos` | `(25, 3)` | 否，参与手部重定向 |
| `left_hand_rot` | `(25, 3, 3)` | 否 |
| `right_hand_rot` | `(25, 3, 3)` | 否 |
| `left_hand_pinch` | `bool` | 否 |
| `right_hand_pinch` | `bool` | 否 |
| `left_hand_pinchValue` | `float` | 否，`dex1 + hand` 用于夹爪控制 |
| `right_hand_pinchValue` | `float` | 否，`dex1 + hand` 用于夹爪控制 |
| `left_hand_squeeze` | `bool` | 否 |
| `right_hand_squeeze` | `bool` | 否 |

### 13.2 controller tracking 下 XR 能拿到

| 字段 | 形状/类型 | 当前是否直接写入 JSON |
| --- | --- | --- |
| `head_pose` | `(4, 4)` | 否 |
| `left_wrist_pose` | `(4, 4)` | 否，参与 IK |
| `right_wrist_pose` | `(4, 4)` | 否，参与 IK |
| `left_ctrl_trigger` | `bool` | 否 |
| `left_ctrl_triggerValue` | `float` | 否，`dex1 + controller` 用于夹爪控制 |
| `left_ctrl_squeeze` | `bool` | 否 |
| `left_ctrl_aButton` | `bool` | 否 |
| `left_ctrl_bButton` | `bool` | 否，`controller + brainco` 用于左手固定食指姿态控制 |
| `left_ctrl_thumbstick` | `bool` | 否，用于双摇杆急停判断 |
| `left_ctrl_thumbstickValue` | `(2,)` | 否，`controller + dex1` 下生成 body 动作 |
| `right_ctrl_triggerValue` | `float` | 否，`dex1 + controller` 用于夹爪控制 |
| `right_ctrl_aButton` | `bool` | 否，用于退出遥操作 |
| `right_ctrl_bButton` | `bool` | 否，`controller + brainco` 用于右手固定食指姿态控制 |
| `right_ctrl_thumbstickValue` | `(2,)` | 否，`controller + dex1` 下生成 body 动作 |

如果后续需要训练模型直接使用头姿、腕部 SE(3)、25 点手部轨迹、pinch/squeeze 或手柄按钮，需要扩展 `states` 或新增 `observations` 字段；当前 `data.json` 不保存这些原始 XR 字段。

## 14. 针对车内物理按键采集的建议模式

当前任务是车内物理按键按压测试，使用 BrainCo 灵巧手。如果使用手势追踪，建议使用：

```bash
python teleop_hand_and_arm.py --record
```

如果使用手柄 B/Y 键触发固定食指姿态，建议使用：

```bash
python teleop_hand_and_arm.py --input-mode controller --ee brainco --record
```

或者显式配置：

```bash
python teleop_hand_and_arm.py \
  --input-mode hand \
  --ee brainco \
  --frequency 15 \
  --task-name vehicle_physical_button_press \
  --record
```

该模式每帧核心监督数据为：

```text
images:
  color_0/color_1: head camera
  color_2: left wrist camera
  color_3: right wrist camera

robot state:
  states.left_arm.qpos
  states.right_arm.qpos
  states.left_ee.qpos
  states.right_ee.qpos

action:
  actions.left_arm.qpos
  actions.right_arm.qpos
  actions.left_ee.qpos
  actions.right_ee.qpos
```

对于“按键按压”这类任务，通常最关键的是：

- `colors`：判断手指、按键和接触位置
- `actions.right_ee.qpos` 或 `actions.left_ee.qpos`：学习手指闭合/按压目标
- `states.right_ee.qpos` 或 `states.left_ee.qpos`：反馈实际手指状态
- `actions.right_arm.qpos` 或 `actions.left_arm.qpos`：学习手腕和手臂靠近按键的轨迹
