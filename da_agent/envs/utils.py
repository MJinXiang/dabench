import signal
import os
import hashlib
import shutil
from typing import Dict
import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
import yaml
import sys
import threading


# TIMEOUT_DURATION = 25
TIMEOUT_DURATION = 40

def is_file_valid(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            pd.read_csv(file_path)
        elif ext == '.json':
            with open(file_path, 'r') as f:
                json.load(f)
        elif ext == '.xml':
            ET.parse(file_path)
        elif ext == '.yaml' or ext == '.yml':
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
        else:
            return True, None
        return True, None
    except Exception as e:
        return False, str(e)
        
class timeout:
    def __init__(self, seconds=TIMEOUT_DURATION, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

        self.timer = None

    def handle_timeout(self, signum=None, frame=None):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        # signal.signal(signal.SIGALRM, self.handle_timeout)
        # signal.alarm(self.seconds)
        
        if sys.platform != 'win32':
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        else:
            # 创建一个包装函数，不带参数
            def wrapper():
                self.handle_timeout()
            
            self.timer = threading.Timer(self.seconds, wrapper)
            self.timer.start()
            # self.timer = threading.Timer(self.seconds, self.handle_timeout)
            # self.timer.start()

    def __exit__(self, type, value, traceback):
        # # signal.alarm(0)
        # if sys.platform != 'win32':
        #     signal.alarm(0)  # Cancel the alarm
        # else:
        #      if self.timer:
        #             self.timer.cancel()  # 取消定时器
        #     # self.timer.cancel()  # Cancel the timer

        try:
            if sys.platform != 'win32':
                signal.alarm(0)  # 取消信号报警
            else:
                if self.timer:
                    self.timer.cancel()  # 取消定时器
        except TimeoutError as e:
            print(f"Caught TimeoutError: {e}")
            return True  # 返回 True 表示异常已被处理


def delete_files_in_folder(folder_path):
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_sha256(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
        return hashlib.sha256(file_data).hexdigest()


