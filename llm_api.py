import os
import inspect
import json
from openai import OpenAI

# 导入key
with open(os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), "./keys.json"), "r") as f:
    KEYS = json.load(f)
OPENAI_KEY = KEYS["openai_key_from_lwd2hzhp"]


def prompt_chat(model_name, prompt, **kwargs):
    client = OpenAI(api_key=OPENAI_KEY)
    messages = [{"role": "user", "content": prompt}]

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )

    return completion.choices[0].message
