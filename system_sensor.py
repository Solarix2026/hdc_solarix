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
from pynput import keyboard as keyboard_pynput
from plyer import notification

import keyboard
from memory_vault import MemoryVault
from hdc_coder import HDCCoder


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
        
        # 抛弃重的 Qwen，采用轻量级 HDCCoder 
        self.hdc_encoder = HDCCoder()
        
        # 状态机：跟踪窗口切换
        self.current_window = self._get_active_window_title()
        self.previous_window = None
        
        # 时间与指标追踪
        self.window_start_time = time.time()
        
        # 滑动窗口：记录最近有效击键的绝对时间戳
        self.keystroke_timestamps = deque()
        
        # 休息与状态干预的打点控制 (双轨制干预核心)
        self.last_intervention_time = time.time()
        self.last_state_eval_time = time.time()
        self.deep_work_start_time = None
        
        # RLHF 反馈存储：key=场景特征向量哈希，value={threshold: 动态阈值, feedback: [0/1]}
        self.rlhf_feedback = {}
        self.last_similar_memory = None  
        self.last_echo_time = 0.0  
        self.dnd_mode = False  
        self.feedback_lock = threading.Lock()  
        
        # 凌晨2点折叠标志
        self.consolidation_done_today = False
        
        # 生产者 - 消费者 异步队列设置
        self.task_queue = queue.Queue()
        
        # UI 弹窗队列：由 Worker 计算触发，交还主线程安全执行，设置最大长度防溢出
        self.notification_queue = queue.Queue(maxsize=10)
        self.is_running = False
        
        # 启动后台消费者线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        
        # 定义并初始化键盘监听器 (生产者)
        self.listener = keyboard_pynput.Listener(on_press=self._on_press)
        
        # 注册免打扰快捷键
        keyboard.add_hotkey('ctrl+alt+s', self._toggle_dnd_mode)

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

    def _toggle_dnd_mode(self):
        """热键触发免打扰模式（Ctrl+Alt+S）"""
        self.dnd_mode = not self.dnd_mode
        print(f"[Solarix] 免打扰模式已{'开启' if self.dnd_mode else '关闭'}")
        if self.dnd_mode:
            # 记录免打扰触发时的场景，后续动态调整该场景的干预频率
            current_window = self._get_active_window_title()
            current_hv = self.hdc_encoder.encode(current_window)
            hv_hash = hash(str(current_hv))
            with self.feedback_lock:
                self.rlhf_feedback[hv_hash] = {"threshold": 1.0, "feedback": [0]}

    def _submit_notify_task(self, task):
        try:
            self.notification_queue.put(task, block=False)
        except queue.Full:
            # 溢出时丢弃最旧的任务，避免阻塞
            try:
                self.notification_queue.get_nowait()
            except queue.Empty:
                pass
            self.notification_queue.put(task, block=False)
            
    def _is_echo_allowed(self) -> bool:
        """语义共鸣提醒节流：30分钟内仅触发一次"""
        if self.dnd_mode:
            return False
        return time.time() - self.last_echo_time > 1800  # 30分钟

    def _show_feedback_dialog_async(self, title, message, hv_hash):
        """异步显示带反馈的对话框（Daemon线程，不阻塞核心循环）"""
        import threading
        dialog_thread = threading.Thread(
            target=self._show_feedback_dialog,
            args=(title, message, hv_hash),
            daemon=True
        )
        dialog_thread.start()

    def _show_feedback_dialog(self, title, message, hv_hash):
        """实际弹窗逻辑（运行在独立Daemon线程），自主处理RLHF反馈"""
        import win32api, win32con
        # 弹窗按钮：确认（接受提醒=1）/取消（拒绝提醒=0）/查看（调起历史解决方案=2）
        resp = win32api.MessageBox(0, message, title, win32con.MB_YESNOCANCEL)
        
        with self.feedback_lock:
            if hv_hash not in self.rlhf_feedback:
                self.rlhf_feedback[hv_hash] = {"threshold": 0.7, "feedback": []}
                
            if resp == 6:  # YES -> 接受提醒
                self.rlhf_feedback[hv_hash]["feedback"].append(1)
                self.rlhf_feedback[hv_hash]["threshold"] = max(0.5, self.rlhf_feedback[hv_hash]["threshold"] - 0.05)
            elif resp == 7:  # NO -> 拒绝提醒
                self.rlhf_feedback[hv_hash]["feedback"].append(0)
                self.rlhf_feedback[hv_hash]["threshold"] = min(0.9, self.rlhf_feedback[hv_hash]["threshold"] + 0.05)
            elif resp == 2 and self.last_similar_memory:  # CANCEL -> 查看记录
                # 预留查看动作，比如打开一个文件
                print(f"[Solarix] 用户选择查看记录，历史方案: {self.last_similar_memory.get('solution')}")
                # self._open_history_solution(self.last_similar_memory["solution_path"])

    def _get_recent_keystrokes(self) -> int:
        """
        计算最近60秒的击键数（滑动窗口的核心防抖逻辑）
        这使得“每分钟敲击数”变成了真实的瞬时速率，而非长时间挂机的被平均数。
        """
        current_time = time.time()
        # 清理60秒前的过期时间戳（维护实时窗口）
        while self.keystroke_timestamps and self.keystroke_timestamps[0] < current_time - 60:
            self.keystroke_timestamps.popleft()
            
        return len(self.keystroke_timestamps)

    def _on_press(self, key):
        """处理键盘按下事件的回调"""
        try:
            # 记录常规字符 (过滤掉修饰键如 Ctrl, Shift 等无 char 属性的内容)
            if hasattr(key, 'char') and key.char is not None:
                self.buffer.append(key.char)
                self.keystroke_timestamps.append(time.time())
            # 处理退格键 (Backspace): 如果缓冲区有内容，则弹掉最后一个字符
            elif key == keyboard_pynput.Key.backspace:
                if self.buffer:
                    self.buffer.pop()
            # 常见的空白控制字符进行可读性保留
            elif key == keyboard_pynput.Key.space:
                self.buffer.append(' ')
                self.keystroke_timestamps.append(time.time())
            elif key == keyboard_pynput.Key.enter:
                self.buffer.append('\n')
                self.keystroke_timestamps.append(time.time())
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
            return
            
        current_time = time.time()
        dwell_time_seconds = current_time - self.window_start_time
        
        # 获取当前最真实的极瞬时敲击频率
        real_keystrokes_1min = self._get_recent_keystrokes()
        
        # 如果是窗口切换（非强制评估），则重置状态机为新窗口做准备
        if not force:
            self.window_start_time = current_time

        # 获取在打字过程中的目标窗口名
        window_title = self.current_window

        # 检查是否为敏感窗口
        if self._is_sensitive_window(window_title):
            key_buffer_str = "<SENSITIVE_CONTENT_FILTERED>"
        else:
            # 锁定当前缓冲区内容并将其提取出来作为本次快照
            key_buffer_str = "".join(self.buffer)
        
        # 如果是窗口切换，清空内部键盘缓冲区留给新窗口
        # 注: 不清空 keystroke_timestamps，因为它是一个全局的时间滑轨，不随窗口销毁
        if not force:
            self.buffer.clear()
        
        # 将结构化原始数据推入消费队列
        self.task_queue.put((current_time, window_title, key_buffer_str, real_keystrokes_1min, dwell_time_seconds, force))


    def _worker_loop(self):
        """后台线程(消费者)：负责执行 HDC 编码、记忆存库与语义共鸣提醒"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1)
                current_time, window_title, key_buffer_str, keystrokes_1min, dwell_time_seconds, is_force_eval = task
                
                # -------------------------------------------------------------
                # 核心重构：100% 提取与存库
                # -------------------------------------------------------------
                if not key_buffer_str.strip() and is_force_eval:
                    self.task_queue.task_done()
                    continue
                    
                current_context = f"[{window_title}] {key_buffer_str}"
                
                # 特征提取
                current_hv = self.hdc_encoder.encode(current_context)
                hv_hash = hash(str(current_hv))
                
                # 入库记忆穹顶
                inserted_id = self.vault.add_memory(
                    hv=current_hv,
                    context=current_context,
                    window_title=window_title,
                    timestamp=current_time
                )
                print(f"[Worker] HDC 记忆已归档 (ID={inserted_id}) - {window_title[:20]}... [驻留 {dwell_time_seconds:.1f}s]")
                
                # -------------------------------------------------------------
                # 语义共鸣提醒触发机制
                # -------------------------------------------------------------
                if self._is_echo_allowed():
                    # 优先通过动态阈值检索
                    with self.feedback_lock:
                        dynamic_threshold = self.rlhf_feedback.get(hv_hash, {}).get("threshold", 0.7)
                        
                    similar_memories = self.vault.retrieve_by_similarity(
                        hv=current_hv, top_k=1, threshold=dynamic_threshold
                    )
                    
                    if similar_memories:
                        self.last_similar_memory = similar_memories[0]
                        sim_score = self.last_similar_memory["similarity_score"]
                        history_context = self.last_similar_memory["context"]
                        history_solution = self.last_similar_memory["solution"]
                        ts = self.last_similar_memory['timestamp']
                        
                        dt_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        remind_msg = f"""Solarix · 记忆共鸣提醒（相似度 {sim_score:.2f}）：
                        
检测到你正在处理「{window_title}」相关场景，
你在 {dt_str} 曾处理过高度相似的问题：
{history_context[:50]}...

当时的解决方案：{history_solution[:100]}...

是否需要调起完整记录？
"""
                        self._show_feedback_dialog_async(
                            title="Solarix · 语义共鸣",
                            message=remind_msg,
                            hv_hash=hv_hash
                        )
                        self.last_echo_time = time.time()
                
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n[Worker] 消费者执行异常: {e}")

    def _memory_consolidation(self):
        """凌晨2点执行记忆折叠：基于HDC多数表决+DBSCAN Hamming距离合并高相似度冗余向量，不删除任何数据"""
        current_hour = datetime.now().hour
        if current_hour == 2 and not self.consolidation_done_today:
            # 1. 检索所有未折叠的记忆向量
            unconsolidated_memories = self.vault.get_unconsolidated_memories()
            if len(unconsolidated_memories) < 5:  # 不足5条不合并
                self.consolidation_done_today = True
                return
            # 2. 解包HDC二进制向量（关键：转换为0/1数组，适配Hamming距离）
            # 原mem["hv"]是packbits压缩的uint8数组，先解包为bool/int型0/1数组
            import numpy as np
            unpacked_vectors = []
            for mem in unconsolidated_memories:
                unpacked_hv = np.unpackbits(mem["hv"]).astype(np.uint8)  # 解包为0/1数组
                unpacked_vectors.append(unpacked_hv)
            unpacked_vectors_array = np.array(unpacked_vectors)  # shape: (n_samples, n_bits)
            
            # 3. 按相似度聚类（HDC二进制向量专用：Hamming距离）
            from sklearn.cluster import DBSCAN
            print("[Solarix] 处于凌晨护城河，开始启动 HDC DBSCAN 记忆折叠...")
            clustering = DBSCAN(
                eps=0.1, 
                min_samples=5, 
                metric='hamming'  # 强制使用Hamming距离，适配二进制向量
            ).fit(unpacked_vectors_array)
            
            # 4. 合并聚类后的向量（严格遵循HDC多数表决定律）
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:  # 噪声点不合并
                    continue
                cluster_mems = [unconsolidated_memories[i] for i, lbl in enumerate(clustering.labels_) if lbl == cluster_id]
                # 5. 提取聚类内所有HDC超向量并解包（关键：先unpackbits）
                cluster_hvs = [np.unpackbits(mem["hv"]) for mem in cluster_mems]
                cluster_hvs_array = np.array(cluster_hvs)  # shape: (n_samples, n_bits)
                # 6. HDC多数表决：按列求和，>半数设为1，≤半数设为0（处理平局）
                sum_bits = np.sum(cluster_hvs_array, axis=0)
                majority_threshold = len(cluster_mems) / 2
                merged_bits = np.where(sum_bits > majority_threshold, 1, 0)
                # 7. 平局处理：随机设为1（HDC允许微小噪声）
                tie_mask = (sum_bits == majority_threshold)
                merged_bits[tie_mask] = np.random.choice([0,1], size=np.sum(tie_mask))
                # 8. 重新打包为uint8型HDC超向量（关键：packbits）
                merged_hv = np.packbits(merged_bits.astype(np.uint8))
                # 9. 存入合并后的记忆，标记原记忆为「已折叠」（不删除）
                self.vault.add_consolidated_memory(
                    merged_hv=merged_hv,
                    context_summary=f"合并{len(cluster_mems)}条相似操作记录",
                    original_ids=[mem["id"] for mem in cluster_mems],
                    timestamp=time.time()
                )
                self.vault.mark_as_consolidated([mem["id"] for mem in cluster_mems])
                print(f"[Solarix] 成功折叠聚簇 {cluster_id}，包含 {len(cluster_mems)} 条记忆。")
            self.consolidation_done_today = True
        elif current_hour != 2:
            self.consolidation_done_today = False

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
                # 只在当前主频上安全弹出通知
                try:
                    toast = self.notification_queue.get_nowait()
                    notification.notify(
                        title=toast.get("title", "系统提示"),
                        message=toast.get("message", ""),
                        app_name="Solarix",
                        timeout=toast.get("timeout", 10)
                    )
                    self.notification_queue.task_done()
                except queue.Empty:
                    pass

                # 快速轮询以降低延迟且节省 CPU 空转
                time.sleep(0.1)
                
                current_time = time.time()
                
                # 预案 1: 每隔一段固定时间 (这里暂设为测试的 60 秒，真实场景可设为 300) 强制推状态给消费者评估
                if current_time - self.last_state_eval_time > 60:
                    self._sample_and_save(force=True)
                    self.last_state_eval_time = current_time
                    # 借用这个1分钟一次的钩子执行低频维护：尝试合并记忆
                    self._memory_consolidation()
                
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