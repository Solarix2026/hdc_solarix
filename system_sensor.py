# C++ Future: This layer stays in Python. It's all about OS APIs, not math.
"""
Solarix — 全天候感知层 (System Sensor)
持续监控用户的操作（键盘输入、活动窗口），并在特定周期将其转化为超维记忆存入底层的MemoryVault。
"""

from __future__ import annotations

import time
import win32gui
from collections import deque
from datetime import datetime
from pynput import keyboard

from memory_vault import MemoryVault
from lsh_mapper import LSHMapper
from qwen_embedder import get_embedding


def get_active_window_title() -> str:
    """获取当前处于前台焦点的活动窗口的标题"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        return title if title else "Unknown Window"
    except Exception:
        return "Unknown Window"


class SystemSensor:
    """全天候感知器，监听用户上下文操作并定期存入知识库。"""

    def __init__(self, vault: MemoryVault, buffer_size: int = 500):
        """
        初始化感知器。
        
        Parameters
        ----------
        vault : MemoryVault
            对应的底层数字海马体（存储库）。
        buffer_size : int
            键盘缓冲区的最大字符长度，超出时自动丢弃最旧字符。
        """
        self.vault = vault
        # 使用 deque 的 maxlen 特性，一旦达到上限自动从另一端淘汰旧数据 (FIFO)
        self.buffer = deque(maxlen=buffer_size) 
        
        # 预先拉起 LSHMapper
        self.mapper = LSHMapper(input_dim=896, output_dim=10000, seed=42)
        
        # 定义并初始化键盘监听器
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.is_running = False

    def _on_press(self, key):
        """处理键盘按下事件的回调"""
        try:
            # 记录常规字符 (过滤掉修饰键如 Ctrl, Shift 等无 char 属性的内容)
            if hasattr(key, 'char') and key.char is not None:
                self.buffer.append(key.char)
            # 对常见的空白控制字符进行可读性保留
            elif key == keyboard.Key.space:
                self.buffer.append(' ')
            elif key == keyboard.Key.enter:
                self.buffer.append('\n')
        except Exception:
            pass

    def _sample_and_save(self):
        """执行一次感知采样、超维哈希并写入记忆库"""
        # 如果缓冲区为空，直接跳过以省电
        if not self.buffer:
            return

        # 锁定当前缓冲区内容并清空原始 deque（提取出来作为本次快照）
        key_buffer_str = "".join(self.buffer)
        self.buffer.clear()

        # 生成感知元数据
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        window_title = get_active_window_title()

        perception_text = f"[{timestamp}] Window: {window_title}\nTyped: {key_buffer_str}"

        # Embedding 和 LSH 降维映射
        hv = self.mapper.map(get_embedding(perception_text))

        # 写入数字海马体
        inserted_id = self.vault.add_memory(perception_text, hv, source="system_sensor")
        print(f"\n[Sensor] 记忆已存档 (ID={inserted_id}) - 前台窗口源: {window_title[:20]}...")

    def start(self, interval_seconds: float = 300.0):
        """
        开启感知守护循环。
        """
        self.is_running = True
        self.listener.start()
        
        try:
            while self.is_running:
                # 采用小分片 sleep 以便能够更快响应 KeyboardInterrupt (Ctrl+C)
                # 这样不会因为一个死长的 300 秒而陷入锁死状态
                for _ in range(int(interval_seconds)):
                    time.sleep(1)
                
                # 时间到，执行一次采样抓取
                self._sample_and_save()
                
        except KeyboardInterrupt:
            # 监听到 Ctrl+C
            self.stop()

    def stop(self):
        """安全停止感知器，并对残余缓冲区做最后存盘。"""
        if not self.is_running:
            return
            
        print("\n正在停止感知器...")
        self.is_running = False
        self.listener.stop()
        
        # 优雅退出前，确保把用户刚打字但还没来得及由定时器触发的残存内容存下来
        if self.buffer:
            self._sample_and_save()
            
        print("感知器已安全退出。")


# ======================================================================
# 测试验证
# ======================================================================
if __name__ == "__main__":
    import os
    
    # 获取环境变量保证无网时 Qwen fallback 加载
    os.environ["HF_HUB_ENABLE_XET"] = "0"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print("=" * 60)
    print("准备启动 Solarix 全天候系统感知层 ...")
    print("=" * 60)

    # 1. 挂载持久化组件
    test_db_path = "solarix_memory.db"
    vault = MemoryVault(db_path=test_db_path)
    
    # 2. 挂载感知器层
    sensor = SystemSensor(vault=vault, buffer_size=500)
    
    print("Solarix 感知器已启动，正在记录您的操作... (按 Ctrl+C 停止)")
    print("提示: 默认时间循环设定在 300秒。我们在真实场景下后台运行。")
    print("(等待输入及时间轮训...)")
    
    try:
        # 这里为测试设置了真实的 300 秒轮询（若你想加快触发可以手动在这里调参，例如 interval_seconds=10）
        sensor.start(interval_seconds=300)
    except KeyboardInterrupt:
        pass
    finally:
        vault.close()