from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from unittest.mock import patch 
load_dotenv()
# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("api_key"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("azure_endpoint")  # Replace with your Azure endpoint
)

def get_translation(post: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful translator that accurately translates text from other languages to English. 
                    Your answer must only be the final translated output."""
                },
                {
                    "role": "user",
                    "content": post
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def get_language(post: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that accurately detects the language of the text provided. 
                    Your answer must only be the name of the language detected in English. 
                    If you can't detect the language or if there are multiple languages, return English as the answer."""
                },
                {
                    "role": "user",
                    "content": post
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Language detection error: {e}")
        return ""

def translate_content(content: str) -> tuple[bool, str]:
    is_english, result = query_llm_robust(content)
    return is_english, result


def query_llm_robust(post: str) -> tuple[bool, str]:
    # Existing input validation
    if not isinstance(post, str):
        return (True, str(post))
    
    if not post.strip():
        return (True, "")
    
    # Check for special characters
    char = set('!@#$%^&*()_+-=[]{}|\\:;"\'<>,.?/~`₹€£¥©®™°×÷§¶⌘⌥⇧⌃☆★')
    text_chars = [c for c in post if c not in char and not c.isspace()]
    if len(text_chars) < len(post.strip()) * 0.5:
        print("c")
        return (True, post)
    
    # Check for numbers
    if any(c.isdigit() for c in post) and sum(c.isdigit() for c in post) > len(post) * 0.3:
        return (True, post)
    
    try:
        detected_language = get_language(post)
        print(detected_language)
        
        # Handle if API returns None or empty response
        if not detected_language or not detected_language.strip():
            return (True, post)
            
        detected_language = detected_language.strip().lower()
        
        if detected_language == "english":
            return (True, post)
        
        
        # List of content filter responses
        error_responses = [
            "ResponsibleAIPolicyViolation",
            "Error code: 400",
            "policy violation",
            "error",
            "unknown",
            "invalid",
            "don't understand",
            "Please modify your prompt and retry."
        ]
        
        # Handle unexpected response
        if any(phrase in str(detected_language) for phrase in error_responses):
            return (True, post)
            
    except Exception:
        return (True, post)
    
    try:
        translation = get_translation(post)
        # Handle if API returns None or empty response
        if not translation or not translation.strip():
            return (True, post)
        translation = translation.strip()
        
    except Exception:
        return (True, post)
    
    # Return the translation result
    return (False, translation)