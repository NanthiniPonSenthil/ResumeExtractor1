import os
import json
from openai import AzureOpenAI

# Configuration
OPENAI_API_KEY = "xyz"
OPENAI_ENDPOINT = "xyz"

# Initialize OpenAI client
client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
    api_version="2024-12-01-preview"
)

# Test skills data
test_skills = ["Python", "Java", "C#", "AWS", "Azure", "Docker", "React", "Angular"]

try:
    print("Testing OpenAI embeddings...")
    
    # Create embedding
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=test_skills
    )
    
    print(f"✅ Success! Generated {len(response.data)} embeddings")
    
    for i, item in enumerate(response.data):
        print(f"Skill: {test_skills[i]}, Embedding length: {len(item.embedding)}")
        print(f"First 5 values: {item.embedding[:5]}")
    
    print(f"Usage: {response.usage}")

except Exception as e:
    print(f"❌ Error: {e}")