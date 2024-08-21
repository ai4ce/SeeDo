import base64
import requests
import os

# OpenAI API Key
api_key = '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'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Directory containing the images
directory_path = "media/charge laptop"

# Get all image paths
image_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Encode all images to base64
base64_images = [encode_image(path) for path in image_paths]


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

question = '''This picture set is a human demonstration of a task. organize your answer in the following format.
Description of task: description
Plan decomposition: decompose the task into a few executable sub-plans
Object inhand: the object to hold in hand
Object unattached: the object to touch
Pre-touch point: the touching part of object inhand
Post-touch point: the touching part of object unattached'''

# Construct messages payload with all images
messages_payload = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"I want you to answer question: {question}"
            }
        ]
    }
]

# Add all images to the payload
for b64_img in base64_images:
    image_message = {
        "role": "system",
        "content": {
            "type": "image",
            "data": f"data:image/png;base64,{b64_img}"
        }
    }
    messages_payload.append(image_message)

# Final payload
payload = {
    "model": "gpt-4-1106-vision-preview",
    "messages": messages_payload,
    "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())