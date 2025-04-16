from openai import OpenAI
api_key = "sk-4641fc1a8560499da62c883dd3472e0c"  # Replace with your actual API key
base_url = "https://api.deepseek.com"  # Replace with your actual base URL
client = OpenAI(api_key=api_key,base_url=base_url)
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)
print(response.choices[0].message.content)