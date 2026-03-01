import time
from memory_vault import MemoryVault
from system_sensor import SystemSensor
import threading

import os
if os.path.exists("temp_sim.db"):
    # This might fail on windows if it's open somewhere, but whatever
    try: os.remove("temp_sim.db")
    except: pass

vault = MemoryVault("temp_sim.db")
sensor = SystemSensor(vault)

# manual mock injects
sensor.is_running = True
sensor.task_queue.put((time.time(), "Test Window 1", "test typing 1", 50, 10.0, False))
sensor.task_queue.put((time.time(), "Test Window 2", "test typing 2", 150, 20.0, False))

sensor.last_echo_time = time.time() - 2000 # Make sure echo is allowed

tw = threading.Thread(target=sensor._worker_loop, daemon=True)
tw.start()

# Waiting for tasks to complete
time.sleep(10)
sensor.is_running = False

print(vault.retrieve_all())
