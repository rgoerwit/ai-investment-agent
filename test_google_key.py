import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
    exit(1)

print(f"✅ Found GOOGLE_API_KEY (Length: {len(api_key)})")

# Configure the SDK
try:
    genai.configure(api_key=api_key)
    print("✅ SDK Configured")
except Exception as e:
    print(f"❌ SDK Configuration Failed: {e}")
    exit(1)

# Test 1: List Models (Basic Connectivity)
print("\n--- Test 1: Listing Models ---")
try:
    models = list(genai.list_models())
    print(f"✅ Success! Found {len(models)} models.")
    # Print the first few model names to verify
    for m in models[:3]:
        print(f"   - {m.name}")
except Exception as e:
    print(f"❌ List Models Failed: {e}")

# Test 2: Simple Generation (Functional Test)
print("\n--- Test 2: Simple Generation ---")
try:
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Hello, can you hear me?")
    print(f"✅ Generation Success! Response: {response.text}")
except Exception as e:
    print(f"❌ Generation Failed: {e}")

# Test 3: Embeddings (Specific to the failed tests)
print("\n--- Test 3: Embeddings ---")
try:
    result = genai.embed_content(
        model="models/embedding-001",
        content="This is a test sentence.",
        task_type="retrieval_document",
        title="Embedding Test",
    )
    print(f"✅ Embedding Success! Vector length: {len(result['embedding'])}")
except Exception as e:
    print(f"❌ Embedding Failed: {e}")
