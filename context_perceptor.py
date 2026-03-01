# C++ Future: This heuristic logic can be ported, but Python is fine for now.
"""
Solarix — 认知状态量化层 (Context Perceptor)
将零散的感知元数据（按键频率、停留时间、内容分类）升维并结构化为人类可读的认知状态标签。
"""

from __future__ import annotations
from memory_vault import MemoryVault


class ContextPerceptor:
    """基于规则与强工程化设计的认知状态评估器"""

    def __init__(self, vault: MemoryVault):
        """
        初始化认知状态评估器 (暂作注入框架预留)。

        Parameters
        ----------
        vault : MemoryVault
            底层知识库实例。
        """
        self.vault = vault

    def _categorize_intensity(self, keystrokes: int) -> str:
        """评估过去 60 秒的击键强度"""
        if keystrokes > 200:
            return "HIGH"
        elif 50 <= keystrokes <= 200:
            return "MEDIUM"
        else:
            return "LOW"

    def _categorize_window(self, title: str) -> str:
        """根据关键字简单分类窗口属性"""
        title_lower = title.lower()

        # 工作流软件
        work_keywords = ["code", "vscode", "pycharm", "terminal", "powershell", "cmd"]
        if any(kw in title_lower for kw in work_keywords):
            return "WORK"

        # 娱乐平台
        entertainment_keywords = ["youtube", "reddit", "netflix", "bilibili", "game"]
        if any(kw in title_lower for kw in entertainment_keywords):
            return "ENTERTAINMENT"

        # 浏览器外壳
        browsing_keywords = ["chrome", "edge", "firefox", "safari"]
        if any(kw in title_lower for kw in browsing_keywords):
            return "BROWSING"

        return "OTHER"

    def _categorize_stability(self, dwell_time_seconds: float) -> str:
        """根据在当前窗口的驻留时长评估注意力集中度"""
        if dwell_time_seconds > 600:  # 10 分钟
            return "DEEP"
        elif 120 <= dwell_time_seconds <= 600:  # 2 - 10 分钟
            return "NORMAL"
        else:  # < 2 分钟
            return "SHALLOW"

    def get_state_vector(self, window_title: str, keystrokes_1min: int, dwell_time_seconds: float) -> dict:
        """
        根据底层输入生成工程化的三维状态特征向量 (字典表示)。
        """
        return {
            "activity_intensity": self._categorize_intensity(keystrokes_1min),
            "window_category": self._categorize_window(window_title),
            "context_stability": self._categorize_stability(dwell_time_seconds),
        }

    def get_state_description(self, state_vector: dict, original_title: str = "") -> str:
        """
        将离散的特征词汇融合成便于 LLM 推演和人工阅读的结构化描述。
        """
        stability = state_vector["context_stability"]
        category = state_vector["window_category"]
        intensity = state_vector["activity_intensity"]
        
        # 为了句式通顺，对部分组合进行美化
        scene_map = {
            "WORK": "working session",
            "ENTERTAINMENT": "entertainment session",
            "BROWSING": "web browsing",
            "OTHER": "general activity"
        }
        
        scene = scene_map.get(category, category)
        
        # E.g. "Cognitive State: DEEP WORK (High intensity, Stable context in VS Code)"
        if stability == "DEEP" and category == "WORK":
            header = "DEEP WORK"
        elif stability == "SHALLOW":
            header = "DISTRACTED " + category
        else:
            header = f"{stability} {category}"
            
        desc = (f"Cognitive State: {header} "
                f"({intensity.capitalize()} intensity input, {stability.lower()} focus "
                f"in {scene})")
                
        return desc


# ======================================================================
# 测试验证
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Solarix ContextPerceptor — 认知量化层孤立测试")
    print("=" * 60)
    
    # 建立一个占位 Vault
    perceptor = ContextPerceptor(vault=None)
    
    # 模拟输入 1：深度编程心流
    v1 = perceptor.get_state_vector(
        window_title="main.py - VS Code", 
        keystrokes_1min=300, 
        dwell_time_seconds=900
    )
    d1 = perceptor.get_state_description(v1)
    
    # 模拟输入 2：频繁切换的摸鱼
    v2 = perceptor.get_state_vector(
        window_title="YouTube - Google Chrome", 
        keystrokes_1min=10, 
        dwell_time_seconds=45
    )
    d2 = perceptor.get_state_description(v2)
    
    # 模拟输入 3：常规查询
    v3 = perceptor.get_state_vector(
        window_title="StackOverflow - Edge", 
        keystrokes_1min=120, 
        dwell_time_seconds=300
    )
    d3 = perceptor.get_state_description(v3)

    print("\n[用例 1] 深度编程心流")
    print(v1)
    print("->", d1)
    
    print("\n[用例 2] 碎片化摸鱼散心")
    print(v2)
    print("->", d2)

    print("\n[用例 3] 正常查阅资料")
    print(v3)
    print("->", d3)
    
    print("\n" + "=" * 60)
    if "DEEP WORK" in d1 and "DISTRACTED" in d2:
        print("校验完成 ✓：状态映射判定准确。")
    else:
        print("校验失败 ✗：状态映射不符合预期。")
