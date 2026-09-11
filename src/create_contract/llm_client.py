import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "API_KEY HERE")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing.")

client = Groq(api_key=GROQ_API_KEY)

def generate(
    user_prompt,
    system_prompt=None,
    max_new_tokens=2500,
    temperature=0.1,
):
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("Groq returned an empty response.")

    return content.strip()