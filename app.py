from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variable to store the latest data
# In a real app, you'd save this to a Database (SQL/NoSQL)
stored_data = []

@app.route('/data', methods=['POST'])
def receive_data():
    global stored_data
    content = request.json
    if "readings" in content:
        stored_data = content["readings"]
        print(f"Received {len(stored_data)} points.")
        return jsonify({"message": "Data Received"}), 200
    return jsonify({"message": "Invalid Data"}), 400

@app.route('/fetch', methods=['GET'])
def fetch_data():
    # Your Mobile App calls this endpoint to get the results
    return jsonify(stored_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
