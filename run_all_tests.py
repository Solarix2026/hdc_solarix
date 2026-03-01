# C++ Future: This will be replaced by CMake/CTest.
"""
Solarix 集成测试总开关 (Integration Test Runner)
按顺序自动运行所有模块的单元测试和端到端测试。
任何一步失败（非0返回码或未检测到预期的成功标志）则立即终止并报错。
"""

import os
import subprocess
import sys

def run_test(step: int, total: int, name: str, script: str):
    print(f"[{step}/{total}] Running {name} Test...")
    
    # 继承当前环境变量，并配置 HuggingFace 下载参数（为了 Stage 3 流畅运行）
    env = os.environ.copy()
    env["HF_HUB_ENABLE_XET"] = "0"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 强制让子进程通过 utf-8 处理输入输出流，防止 Windows 控制台默认编码如 cp1252 的崩溃
    env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            env=env,
            capture_output=True
        )
        
        # 强制按 utf-8 手动解码
        stdout_str = result.stdout.decode("utf-8", errors="ignore")
        stderr_str = result.stderr.decode("utf-8", errors="ignore")
        
        # 实时打印子进程的输出，这样能看到测试具体的过程内容
        if stdout_str:
            print(stdout_str)
        if stderr_str:
            print(stderr_str, file=sys.stderr)
            
        # 1. 检查返回码
        if result.returncode != 0:
            print(f"❌ TEST FAILED: 进程返回码不为 0 (code={result.returncode})")
            sys.exit(1)
            
        # 2. 检查输出标志是否达标
        if script == "hdc_core.py" and "回归测试完成 ✓" not in stdout_str:
            print("❌ TEST FAILED: hdc_core.py 中未检测到 '回归测试完成 ✓'")
            sys.exit(1)
        elif script == "lsh_mapper.py" and "验证完成 ✓" not in stdout_str:
            print("❌ TEST FAILED: lsh_mapper.py 中未检测到 '验证完成 ✓'")
            sys.exit(1)
        elif script == "solarix_test_stage3.py":
            # 匹配我们在 Stage3 中设置的健康或警告标志
            if "状态健康" not in stdout_str and "警告" not in stdout_str and "HEALTHY" not in stdout_str and "WARNING" not in stdout_str:
                print("❌ TEST FAILED: Stage 3 中未检测到有关相似度拥挤状态的诊断信息")
                sys.exit(1)
        elif script == "memory_vault.py" and "验证完成 ✓" not in stdout_str:
            print("❌ TEST FAILED: memory_vault.py 中未检测到 '验证完成 ✓'")
            sys.exit(1)
                
    except Exception as e:
        print(f"❌ TEST FAILED: 执行异常 - {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"开始执行 Solarix 自动化集成测试...\n")
    
    # 测试关卡列表
    tests = [
        ("HDCCore", "hdc_core.py"),
        ("LSHMapper", "lsh_mapper.py"),
        ("Stage 3 E2E", "solarix_test_stage3.py"),
        ("MemoryVault", "memory_vault.py")
    ]
    
    total = len(tests)
    
    for i, (name, script) in enumerate(tests, 1):
        if not os.path.exists(script):
            print(f"❌ TEST FAILED: 找不到测试脚本 {script}")
            sys.exit(1)
            
        run_test(i, total, name, script)
        
    # 所有测试顺利通过
    print("============================================================")
    print("🚀 ALL SYSTEMS GO. Integration Test PASSED.")
    print("============================================================")
