import os
import io
import json
import requests
import torch
import dashscope
from dashscope import Generation
from io import BytesIO
from PIL import Image,  ImageChops
from datetime import datetime
import tempfile
import random
import platform
import hashlib
import base64

p = os.path.dirname(os.path.realpath(__file__))

def get_qwenvl_api_key():
    try:
        config_path = os.path.join(p, 'config.json')
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:  
            config = json.load(f)
        api_key = config["QWENVL_API_KEY"]
        print(f"Loaded API key: {api_key[:8]}...")
        return api_key
    except Exception as e:
        print(f"加载配置文件出错: {str(e)}")
        return ""


class QWenChatApi:

    def __init__(self):
        self.api_key = get_qwenvl_api_key()
        if self.api_key is not None:
            dashscope.api_key=self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "你好", "multiline": True}),
                "model_name": (["qwen-max", "qwen-max-1201", "qwen-max-longcontext"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "qwen_generation"

    CATEGORY = "Zho模块组/QWenChatApi"


    def qwen_generation(self, prompt, model_name, seed):
        if not self.api_key:
            raise ValueError("API key is required")

        print(f"API Key: {self.api_key}")
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt}")

        torch.manual_seed(seed)

        try:
            response = Generation.call(
                model=model_name,
                prompt=prompt,
                seed=seed
            )
            print(f"API Response: {response}")
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return ("API调用出错，请检查API key和网络连接", )

        if response is None:
            print("API返回为空")
            return ("No response generated", )

        try:
            if isinstance(response, dict):
                if 'output' in response:
                    output = response['output']
                    if 'text' in output:
                        return (output['text'], )
        except Exception as e:
            print(f"处理API响应时出错: {str(e)}")

        return ("No response generated", )


class QWenChatApiMulti:

    def __init__(self):
        self.api_key = get_qwenvl_api_key()
        self.messages = []  # 初始化对话历史为空
        if self.api_key is not None:
            dashscope.api_key=self.api_key

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "你好", "multiline": True}),
                "model_name": (["qwen-max", "qwen-max-1201", "qwen-max-longcontext"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "qwen_generation"

    CATEGORY = "Zho模块组/QWenChatMultiApi"


    def format_qwchat_history(self):
        formatted_history = []
        for message in self.messages:
            role = message['role']
            contents = message['content']
            for content in contents:
                if 'text' in content:
                    text = content['text']
                    formatted_message = f"{role}: {text}"
                    formatted_history.append(formatted_message)
            formatted_history.append("-" * 40)  # 添加分隔线
        return "\n".join(formatted_history)

    def qwen_generation(self, prompt, model_name, seed):
        if not self.api_key:
            raise ValueError("API key is required")

        # 添加用户消息
        self.messages.append({
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        })

        print(f"Model: {model_name}")
        print(f"Messages: {self.messages}")

        torch.manual_seed(seed)

        try:
            response = dashscope.MultiModalConversation.call(model=model_name, messages=self.messages, seed=seed)
            print(f"API Response: {response}")
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return ("API调用出错，请检查API key和网络连接", )

        if response is None:
            print("API返回为空")
            return ("No response generated", )

        try:
            if isinstance(response, dict):
                if 'output' in response:
                    output = response['output']
                    if 'choices' in output:
                        choices = output['choices']
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get('message', {})
                            if isinstance(message, dict):
                                # 更新对话历史
                                if 'role' in message and 'content' in message:
                                    self.messages.append({
                                        'role': message['role'],
                                        'content': message['content']
                                    })
                                # 获取响应文本
                                content = message.get('content', [])
                                if content and isinstance(content[0], dict):
                                    text = content[0].get('text')
                                    if text:
                                        return (text, )
        except Exception as e:
            print(f"处理API响应时出错: {str(e)}")

        return ("No response generated", )


NODE_CLASS_MAPPINGS = {
    "QWenChatApi": QWenChatApi,
    "QWenChatApiMulti": QWenChatApiMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QWenChatApi": "QWenChatApi",
    "QWenChatApiMulti": "QWenChatApiMulti",
}
