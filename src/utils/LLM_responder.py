# LLM_responder.py
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# ======= CONFIG =======
CHAT_MODEL_4O = "gpt-4o"
CHAT_MODEL_41 = "gpt-4.1"


class LLM_responder:
    """LLM api 物件"""  

    def __init__(self, api_key=None):
        """LLM api建構子"""   
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要設定 OPENAI_API_KEY 環境變數或直接傳入 API 金鑰")
        
        # 初始化 OpenAI 客戶端
        self.client = OpenAI(api_key=self.api_key)

    def __call_completions_openai_api(self, model, messages, temperature = 0.7, top_p = 1):
        """使用 OpenAI 套件呼叫 chat.completions API"""
        try:
            # 使用 OpenAI 客戶端發送請求
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p = top_p
            )
            
            # 從回應中提取內容
            return response
        except Exception as e:
            raise Exception(f"API 請求失敗: {str(e)}")

    def chat_gpt_4o(self, prompt: str, temperature: int)->str:
        """簡單對話-使用 chat.completions API透過gpt-4o的user回答單輪對話"""
        model = CHAT_MODEL_41
        prompt = [{"role": "user", "content": prompt}]
        respond = self.__call_completions_openai_api(model, prompt, temperature)
        return respond.choices[0].message.content
    
