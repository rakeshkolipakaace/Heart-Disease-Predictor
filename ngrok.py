from pyngrok import ngrok
import os

# Get your ngrok authtoken from Colab secrets or replace 'YOUR_NGROK_AUTHTOKEN' with your actual token
# If you've saved it as a secret named 'NGROK_AUTH_TOKEN'
# NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "YOUR_NGROK_AUTHTOKEN")
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# For direct entry (less secure for sharing notebooks, but works for debugging)
ngrok.set_auth_token("39eyTnJoir6C1bvskKiXxxVCO5p_NeqxYL6tyU2QmemERbjb") # <--- REPLACE THIS WITH YOUR ACTUAL NGROK AUTH TOKEN
# After line 10 in ngrok.py, add:
def start_ngrok_tunnel(port=8501):
    """Start ngrok tunnel for Streamlit app"""
    # Kill existing tunnels
    ngrok.kill()
    
    # Create new tunnel
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel available at: {public_url}")
    return public_url