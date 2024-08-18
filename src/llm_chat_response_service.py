import requests
from dotenv import dotenv_values

config = dotenv_values('.env')
openai_api_key = config["OPENAI_API_KEY"]  # Ensure this is set securely

def get_chat_response(message, model="gpt-4o-2024-08-06", max_tokens=2000, temperature=0.3, stream=True):
    url = "https://api.openai.com/v1/files"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    print(payload, headers)
    response = requests.post(url, json=payload, headers=headers)
    print(response.content)
    return response

"""     if response.status_code == 200:
        content = ""
        content_parts = response.content.decode('utf-8').split("\r\n\r\n")
        for part in content_parts:
            try:
                if part.strip() and part[6:] is not None:
                    json_part = json.loads(part[6:])
                    delta = json_part['choices'][0]['delta']

                    content += delta.get('content', '')
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
        return content
    else:
        return f"Request failed with status code {response.status_code}" """
