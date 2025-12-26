import requests, json
from sseclient import SSEClient

SPACE = "https://0504ankitsharma-glucoma.hf.space"

json_input = '[{"t":0,"v":512},{"t":12,"v":530},{"t":25,"v":548}]'
payload = {"data": [json_input]}

# 1) POST -> event_id
post = requests.post(f"{SPACE}/gradio_api/call/predict", json=payload, timeout=30)
print("POST:", post.status_code, post.text)
post.raise_for_status()
event_id = post.json()["event_id"]

# 2) GET -> SSE stream
stream_url = f"{SPACE}/gradio_api/call/predict/{event_id}"
print("STREAM URL:", stream_url)

resp = requests.get(stream_url, stream=True, timeout=60)
resp.raise_for_status()

client = SSEClient(resp)

for msg in client.events():   # ✅ important
    print("EVENT:", msg.event, "DATA:", msg.data)

    if msg.event in ("complete", "completed"):
        arr = json.loads(msg.data)  # usually ["result"]
        print("\n✅ FINAL RESULT:", arr[0] if isinstance(arr, list) and arr else arr)
        break

    if msg.event == "error":
        print("\n❌ ERROR:", msg.data)
        break
