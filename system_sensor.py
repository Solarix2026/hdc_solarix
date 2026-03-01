# C++ Future: This layer stays in Python. It's all about OS APIs, not math.
"""
Solarix — 全天候感知层 (System Sensor)
持续监控用户的操作（键盘输入、活动窗口），并在上下文切换或特定事件发生时将其转化为超维记忆存入底层的MemoryVault。
"""

from __future__ import annotations

import time
import win32gui
import threading
import queue
from collections import deque
from datetime import datetime
from pynput import keyboard
from plyer import notification

from memory_vault import MemoryVault
from lsh_mapper import LSHMapper
from qwen_embedder import get_embedding
from context_perceptor import ContextPerceptor


class SystemSensor:
    """全天候感知器，基于事件驱动、拦截器与后台异步计算监听用户操作并存入知识库。"""

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
        
        # 预先拉起 LSHMapper (将在后台线程调用)
        self.mapper = LSHMapper(input_dim=896, output_dim=10000, seed=42)
        
        # 拉起认知评估器 (Perceptor)
        self.perceptor = ContextPerceptor(vault)
        
        # 状态机：跟踪窗口切换
        self.current_window = self._get_active_window_title()
        self.previous_window = None
        
        # 时间与指标追踪 (统一使用 Unix time float)
        self.window_start_time = time.time()
        self.keystroke_count = 0
        
        # 休息与状态干预的打点控制
        self.last_break_time = time.time()
        self.last_state_eval_time = time.time()
        
        # 生产者 - 消费者 异步队列设置
        self.task_queue = queue.Queue()
        self.is_running = False
        
        # 启动后台消费者线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        
        # 定义并初始化键盘监听器 (生产者)
        self.listener = keyboard.Listener(on_press=self._on_press)

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
                self.keystroke_count += 1
            # 处理退格键 (Backspace): 如果缓冲区有内容，则弹掉最后一个字符
            elif key == keyboard.Key.backspace:
                if self.buffer:
                    self.buffer.pop()
            # 常见的空白控制字符进行可读性保留
            elif key == keyboard.Key.space:
                self.buffer.append(' ')
                self.keystroke_count += 1
            elif key == keyboard.Key.enter:
                self.buffer.append('\n')
                self.keystroke_count += 1
            # 对于删除键 (Delete)，因为目前缺乏光标感知概念，故暂时放行忽略不记
        except Exception:
            pass

    def _sample_and_save(self, force: bool = False):
        """
        主线程(生产者) 调用：仅提取上下文与打字内容并推入队列，绝不包含重型运算阻塞主循环。
        如果 force=True，仅代表时间到了拉取强制评估快照，不重置系统底层状态及清空键盘缓冲区。
        """
        # 如果缓冲区为空并且没有强制拉取指令，直接跳过以省电
        if not self.buffer and not force:
            # 即使跳过，由于窗口也在切换，因此重置计算起始状态以服务于下个空窗口
            self.window_start_time = time.time()
            self.keystroke_count = 0
            return
            
        current_time = time.time()
        dwell_time_seconds = current_time - self.window_start_time
        
        # 捕捉当前的敲击计数值并缓存一份
        strokes = self.keystroke_count
        
        # 如果是窗口切换（非强制评估），则重置状态机为新窗口做准备
        if not force:
            self.window_start_time = current_time
            self.keystroke_count = 0

        # 获取在打字过程中的目标窗口名
        window_title = self.current_window

        # 检查是否为敏感窗口
        if self._is_sensitive_window(window_title):
            key_buffer_str = "<SENSITIVE_CONTENT_FILTERED>"
        else:
            # 锁定当前缓冲区内容并将其提取出来作为本次快照
            key_buffer_str = "".join(self.buffer)
        
        # 如果是窗口切换，清空内部键盘缓冲区留给新窗口
        if not force:
            self.buffer.clear()
        
        # 将结构化原始数据推入消费队列 (包含 is_force_eval 标志，传达最后一位)
        self.task_queue.put((current_time, window_title, key_buffer_str, strokes, dwell_time_seconds, force))


    def _worker_loop(self):
        """后台线程(消费者)：负责执行 Qwen Embedding、认知评估与入库分析等耗时操作"""
        while self.is_running:
            try:
                # 获取任务（设置 timeout 让线程能定期醒来检查 is_running 标志以便安全退出）
                task = self.task_queue.get(timeout=1)
                
                # 解包获取上下文
                current_time, window_title, key_buffer_str, strokes, dwell_time_seconds, is_force_eval = task
                
                # -------------------------------------------------------------
                # 模块一：认知量化与主动干预 (Perceptor Layer)
                # -------------------------------------------------------------
                # 防止除零错误，将击键转换为次/分钟的概念
                keystrokes_1min = (strokes / max(dwell_time_seconds, 1e-6)) * 60.0
                state_vector = self.perceptor.get_state_vector(window_title, keystrokes_1min, dwell_time_seconds)
                state_desc = self.perceptor.get_state_description(state_vector)
                
                print(f"\n[Perceptor] 认知状态: {state_desc}")
                
                time_since_last_break = current_time - self.last_break_time
                
                # 基础干预触发条件: 我们设定为娱乐中，或者工作低频状态（比如走神）
                is_entertainment_or_idle = (state_vector['window_category'] == "ENTERTAINMENT" or 
                                           state_vector['activity_intensity'] == "LOW")
                
                # 为方便测试，这里设定阈值为 60 秒 (真实环境应是 3600 秒)
                if is_entertainment_or_idle and time_since_last_break > 60:
                    # 预案 2：打扰成本评估
                    if keystrokes_1min < 10:
                        print("[Perceptor] 检测到放松状态，但用户可能在专注观看内容 (视频等)，无活跃操作，暂缓干预。")
                        self.last_break_time += 600  # 顺延 10 分钟 (实际测试中也可以降低点)
                    else:
                        print("[Perceptor] 执行系统级硬干预弹窗！")
                        notification.notify(
                            title="Solarix 注意力干预",
                            message="检测到你正在无意义消耗，且已持续运行一段时间。立刻站起来喝水休息。",
                            app_name="Solarix",
                            timeout=10
                        )
                        self.last_break_time = time.time()
                
                # -------------------------------------------------------------
                # 模块二：数据持久化 (HDC Memory Vault)
                # -------------------------------------------------------------
                # 将一致的浮点时间戳转为人类可读字符串存库
                timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
                
                # 对于强制性定时评估拉取的数据不再存入废弃库中占用空间，只更新状态，切屏瞬间才有价值存下。
                if not is_force_eval and key_buffer_str.strip():
                    perception_text = f"[{timestamp_str}] Window: {window_title} (Dwell: {dwell_time_seconds:.1f}s, Keystrokes: {strokes})\nTyped: {key_buffer_str}"
                    
                    # 这部分包含重型神经网络和哈希降维推演 (耗时操作放在后台执行)
                    hv = self.mapper.map(get_embedding(perception_text))
                    
                    # 写入数字海马体
                    inserted_id = self.vault.add_memory(perception_text, hv, source="system_sensor")
                    print(f"[Worker] 记忆异步存档 (ID={inserted_id}) - {window_title[:20]}... [驻留 {dwell_time_seconds:.1f}s | {strokes} 键]")
                
                # 标记该任务为完成
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n[Worker] 消费者执行异常: {e}")

    def start(self):
        """
        开启基于事件(窗口级切换)与高频评估巡检的感知守护循环。
        """
        self.is_running = True
        # 拉起后台消费者线程
        self.worker_thread.start()
        # 拉起键盘拦截器
        self.listener.start()
        
        print(f"[*] 初始焦点窗口: {self.current_window[:40]}...")
        
        try:
            while self.is_running:
                # 快速轮询以降低延迟且节省 CPU 空转
                time.sleep(0.1)
                
                current_time = time.time()
                
                # 预案 1: 每隔一段固定时间 (这里暂设为测试的 60 秒，真实场景可设为 300) 强制推状态给消费者评估
                if current_time - self.last_state_eval_time > 60:
                    self._sample_and_save(force=True)
                    self.last_state_eval_time = current_time
                
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