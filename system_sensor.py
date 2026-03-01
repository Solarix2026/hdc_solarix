# C++ Future: This layer stays in Python. It's all about OS APIs, not math.
"""
Solarix — 全天候感知层 (System Sensor)
持续监控用户的操作（键盘输入、活动窗口），并在上下文切换或特定事件发生时将其转化为超维记忆存入底层的MemoryVault。
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


class SystemSensor:
    """全天候感知器，基于事件驱动和上下文切换监听用户操作并存入知识库。"""

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
        
        # 状态机：跟踪窗口切换
        self.current_window = self._get_active_window_title()
        self.previous_window = None
        
        # 定义并初始化键盘监听器
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.is_running = False

    def _get_active_window_title(self) -> str:
        """获取当前处于前台焦点的活动窗口的标题"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            return title if title else "Unknown Window"
        except Exception:
            return "Unknown Window"

    def _is_sensitive_window(self, title: str) -> bool:
        """启发式过滤：判断当前窗口标题是否包含敏感词汇（密码、登录等）。"""
        sensitive_keywords = ["login", "sign in", "password", "银行", "登录", "支付宝", "密码"]
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in sensitive_keywords)

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

        # 获取在打字过程中的目标窗口名 (即产生这段敲击动作原本所在的上下文窗口)
        window_title = self.current_window

        # 检查是否为敏感窗口（登录/网银等），若是，强制掩码替代，防止记录密码或敏感信息
        if self._is_sensitive_window(window_title):
            key_buffer_str = "<SENSITIVE_CONTENT_FILTERED>"
        else:
            # 锁定当前缓冲区内容并将其提取出来作为本次快照
            key_buffer_str = "".join(self.buffer)
        
        # 无论有没有读取，最终清空内部键盘缓冲区留给新窗口
        self.buffer.clear()

        # 生成感知元数据
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        perception_text = f"[{timestamp}] Window: {window_title}\nTyped: {key_buffer_str}"

        # Embedding 和 LSH 降维映射
        hv = self.mapper.map(get_embedding(perception_text))

        # 写入数字海马体
        inserted_id = self.vault.add_memory(perception_text, hv, source="system_sensor")
        print(f"\n[Sensor] 记忆已存档 (ID={inserted_id}) - 前台窗口源: {window_title[:40]}...")

    def start(self):
        """
        开启基于事件(窗口级切换)的感知守护循环。
        """
        self.is_running = True
        self.listener.start()
        
        print(f"[*] 初始焦点窗口: {self.current_window[:40]}...")
        
        try:
            while self.is_running:
                # 快速轮询以降低延迟且节省 CPU 空转
                time.sleep(0.1)
                
                # 获取最新的前台焦点窗口
                new_window = self._get_active_window_title()

                # 判断上下文是否发生了漂移
                if new_window != self.current_window:
                    # 发现存在切换（例如从 VS Code 切到了 Chrome）
                    
                    # 1. 马上结算并保存这离开之前的上个窗口的动作记忆
                    self._sample_and_save()
                    
                    # 2. 从严控盘：不管 _sample_and_save 中是否省电 return，强行清空残留输入避免越界污染
                    self.buffer.clear()
                    
                    # 3. 推进状态机
                    self.previous_window = self.current_window
                    self.current_window = new_window
                    
                    print(f"[Context Shift] {self.previous_window[:30]} -> {self.current_window[:30]}")
                    
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
    print("准备启动 Solarix 全天候系统感知层 (事件驱动版) ...")
    print("=" * 60)

    # 1. 挂载持久化组件
    test_db_path = "solarix_memory.db"
    vault = MemoryVault(db_path=test_db_path)
    
    # 2. 挂载感知器层
    sensor = SystemSensor(vault=vault, buffer_size=500)
    
    print("Solarix 感知器已启动，正在监听您的窗口切换及键盘敲击... (按 Ctrl+C 停止)")
    print("提示：请尝试在当前窗口打字，然后切换到另一个窗口（如浏览器或记事本），观察控制台上下文重定！")
    print("-" * 60)
    
    try:
        sensor.start()
    except KeyboardInterrupt:
        pass
    finally:
        vault.close()