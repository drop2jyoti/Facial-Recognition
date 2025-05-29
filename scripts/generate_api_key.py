import secrets

if __name__ == "__main__":
    api_key = secrets.token_urlsafe(32)
    print(f"Generated API Key: {api_key}") 