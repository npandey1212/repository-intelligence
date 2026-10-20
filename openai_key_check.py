import os
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError
import httpx
def check_openai_api_key_with_request(api_key: str) -> bool:
    """
    Validates an OpenAI API key by sending a minimal text generation request.
    """

    # 1. Create a custom HTTP client with SSL verification disabled
    custom_client = httpx.Client(verify=False)
    client = OpenAI(api_key=api_key,http_client=custom_client)
    
    try:
        # Sends a minimal test prompt using a standard model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Respond with the word: Hello"}],
            max_tokens=5 # Kept low to minimize token cost
        )
        
        # Print the response text
        print(f"✅ Success! OpenAI responded: {response.choices[0].message.content.strip()}")
        return True
        
    except AuthenticationError as e:
        print("❌ Error: Invalid OpenAI API key. Please check your credentials.")
        print(f"Underlying cause: {e.__cause__}") # Shows the connection drop details
        return False
    except RateLimitError:
        print("❌ Error: Key is valid, but you hit a rate limit or ran out of credits.")
        return False
    except APIConnectionError as e:
        print("❌ Error: Could not connect to OpenAI servers. Check your connection.")
        print(f"Underlying cause: {e.__cause__}") # Shows the connection drop details
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":

    # Retrieve key from environment or hardcode for a quick test
    #API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    API_KEY="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"
    if API_KEY == "your-api-key-here" or not API_KEY:
        print("⚠️ Please provide a valid API key string or set the OPENAI_API_KEY environment variable.")
    else:
        check_openai_api_key_with_request(API_KEY)
