"""Fresh image checks for collection; no robot or camera dependencies."""
import math
import time


class ImageStreamGuard:
    VERSION = 'collection-image-guard-v2'
    FRAME_TIMEOUT_SECONDS = 1.0
    STARTUP_TIMEOUT_SECONDS = 5.0
    ERROR_REPEAT_SECONDS = 5.0

    def __init__(self, camera_config, host, logger, is_valid_image):
        self.cameras = {
            name: f"tcp://{host}:{config.get('zmq_port', '?')}"
            for name, config in camera_config.items()
            if isinstance(config, dict) and config.get('enable_zmq', False)
        }
        if not self.cameras:
            raise ValueError('无法录制：没有启用任何 ZMQ 相机。')
        self.logger = logger
        self.is_valid_image = is_valid_image
        self.started_at = time.monotonic()
        self.last_valid = {}
        self.unavailable = list(self.cameras)
        self.faulted = []
        self.details = {}
        self._error_active = False
        self._last_error_signature = None
        self._last_error_at = float('-inf')

    @property
    def ready(self):
        return not self.unavailable

    @property
    def failed(self):
        return bool(self.faulted)

    @property
    def failure_reason(self):
        names = self.faulted or self.unavailable
        return '; '.join(self.details.get(name, name) for name in names)

    def update(self, frames):
        now = time.monotonic()
        unavailable, faulted, details = [], [], {}
        for name, endpoint in self.cameras.items():
            frame = frames.get(name)
            try:
                received_at = float(getattr(frame, 'received_at', None))
            except (TypeError, ValueError):
                received_at = float('nan')
            fresh = (
                math.isfinite(received_at)
                and 0 <= now - received_at < self.FRAME_TIMEOUT_SECONDS
                and self.is_valid_image(getattr(frame, 'bgr', None))
            )
            if fresh:
                self.last_valid[name] = received_at
                continue
            unavailable.append(name)
            last_valid = self.last_valid.get(name)
            if last_valid is None:
                age = now - self.started_at
                is_fault = age >= self.STARTUP_TIMEOUT_SECONDS
                detail = f'{name} ({endpoint}) 尚无有效实时图像，等待 {age:.1f} 秒'
            else:
                age = now - last_valid
                is_fault = age >= self.FRAME_TIMEOUT_SECONDS
                detail = f'{name} ({endpoint}) 已 {age:.1f} 秒没有有效新图像'
            details[name] = detail
            if is_fault:
                faulted.append(name)
        self.unavailable, self.faulted, self.details = unavailable, faulted, details
        if faulted:
            signature = tuple(faulted)
            if (not self._error_active or signature != self._last_error_signature
                    or now - self._last_error_at >= self.ERROR_REPEAT_SECONDS):
                self.logger.error(
                    '[采集图像断流] ' + self.failure_reason
                    + '。图像服务可能已退出或相机异常，录制已禁止；请检查图像服务和网络。'
                )
                self._last_error_at = now
                self._last_error_signature = signature
            self._error_active = True
        elif self.ready and self._error_active:
            self.logger.warning(
                '[采集图像恢复] 全部启用相机已恢复实时图像。异常片段不会续录；'
                '请确认初始条件后重新按 Enter 开始新片段。'
            )
            self._error_active = False

    def wait_until_ready(self, read_frames):
        while True:
            self.update(read_frames())
            if self.ready:
                return
            if time.monotonic() - self.started_at >= self.STARTUP_TIMEOUT_SECONDS:
                raise ConnectionError('无法开始采集：' + self.failure_reason)
            time.sleep(0.033)
