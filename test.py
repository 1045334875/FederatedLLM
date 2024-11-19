import subprocess
import time

# 定义一个函数来执行nvidia-smi命令
def run_nvidia_smi():
    try:
        # 执行nvidia-smi命令并捕获输出
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        # 打印输出结果
        print(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印错误信息
        print("Failed to run nvidia-smi:", e)

# 主循环，每600秒（即10分钟）运行一次nvidia-smi
while True:
    run_nvidia_smi()
    # 等待600秒（10分钟）
    time.sleep(600)