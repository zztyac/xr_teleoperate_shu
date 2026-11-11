# for motion switcher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# for loco client
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import time

# 站立时两膝约 0.6 rad，车内坐姿约 1.5–1.8 rad；两膝都小于该阈值视为站立。
STANDING_KNEE_MAX_RAD = 1.0


def legs_look_standing(leg_q):
    """leg_q: G1 joints 0..11 (left leg 0..5, right leg 6..11); knees are 3 and 9."""
    return max(float(leg_q[3]), float(leg_q[9])) < STANDING_KNEE_MAX_RAD


def read_leg_q(timeout=3.0):
    """Read-only: subscribe rt/lowstate and return the 12 leg joint angles. Publishes nothing."""
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            msg = subscriber.Read(0.5)
            if msg is not None:
                return [msg.motor_state[i].q for i in range(12)]
    finally:
        subscriber.Close()
    raise RuntimeError("No rt/lowstate within %.1fs; is the robot connected?" % timeout)


def debug_mode_refusal(leg_q):
    """Message explaining why debug mode must not be entered, or None when it is safe."""
    if not legs_look_standing(leg_q):
        return None
    return ("机器人看起来是站着的（膝关节 %.2f / %.2f rad）。不加 --motion 会进入 debug 模式并释放运控，"
            "站立的机器人会摔倒。站立采集请加 --motion；确认机器人已吊起时才可加 --allow-debug-while-standing。"
            % (leg_q[3], leg_q[9]))


# MotionSwitcher used to switch mode between debug mode and ai mode
class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None
    
    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None

class LocoClientWrapper:
    def __init__(self):
        self.client = LocoClient()
        self.client.SetTimeout(0.0001)
        self.client.Init()

    def Enter_Damp_Mode(self):
        self.client.Damp()
    
    def Move(self, vx, vy, vyaw):
        self.client.Move(vx, vy, vyaw, continous_move=False)

if __name__ == '__main__':
    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    ms = MotionSwitcher()
    status, result = ms.Enter_Debug_Mode()
    print("Enter debug mode:", status, result)
    time.sleep(5)
    status, result = ms.Exit_Debug_Mode()
    print("Exit debug mode:", status, result)
    time.sleep(2)
