import google.generativeai as genai

genai.configure(api_key="AIzaSyA93cqzdxbddEYoUWoBZk-aUhfk6X4l5xk")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)