# 键盘采集录制与自动回放

`keyboard_macro.py` 是现有键盘采集程序外面的一层终端包装器，不需要修改原来的控制代码。
它会把人工按键原样转发给采集程序，同时记录按键内容和相对时间；回放时再把相同按键按相同节奏发送给采集程序。

## 1. 录制一次标准过程

假设原来的采集命令是：

```bash
python3 your_collect_script.py --your-args
```

改为：

```bash
python3 keyboard_macro.py record macros/button_task.json -- \
  python3 your_collect_script.py --your-args
```

像以前一样完成一次标准操作。结束时按 `Ctrl-]`（先按住 Ctrl，再按 `]`），脚本会保存 JSON 文件。
这个结束键不会发送给机器人，也不会进入录制数据。

如果目标文件已经存在，脚本会拒绝覆盖；确认要更新模板时添加 `--force`。

## 2. 检查录制结果

```bash
python3 keyboard_macro.py show macros/button_task.json
```

输出会列出事件时间和可读按键名。JSON 中的 `data` 是实际回放使用的 Base64 原始终端字节，不建议手工修改。

## 3. 自动回放采集

回放一次：

```bash
python3 keyboard_macro.py replay macros/button_task.json
```

按键文件会记住录制时启动的采集命令，因此通常无需重复填写。也可以在末尾覆盖命令：

```bash
python3 keyboard_macro.py replay macros/button_task.json -- \
  python3 your_collect_script.py --your-new-args
```

连续回放 20 次、每次间隔 3 秒：

```bash
python3 keyboard_macro.py replay macros/button_task.json \
  --repeat 20 --interval 3
```

常用参数：

- `--speed 0.8`：以 0.8 倍速度回放，动作更慢；`--speed 2` 为两倍速度。
- `--start-delay 5`：启动采集程序后等待 5 秒再发送第一个按键。
- `--finish-delay 2`：最后一个按键发送后等待 2 秒。
- `--terminate-after`：全部回放后自动结束采集子进程。默认不结束，方便继续观察或人工接管。
- `Ctrl-]`：任何时候紧急停止自动回放。

## 安全建议

第一次使用新模板时让机器人处于低速、安全空间，仅使用 `--repeat 1` 验证。确保录制内容包含一次任务完整的“开始采集、动作、结束/保存采集”按键，而且起始机械臂状态与录制时一致。涉及视觉定位、接触或物体位置变化的任务，不应只依靠固定时间按键盲回放；应在控制程序中增加状态或传感器检查。
