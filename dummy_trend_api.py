from flask import Flask, jsonify, request

app = Flask(__name__)

# Dummy dataset for any monitor id
DUMMY_DATA = [
    {
        "PROCESS_PARAMETER": {
            "001_A": 40,
            "001_B": 0.3,
            "001_C": 12,
            "001_D": 56,
            "001_E": 850,
            "001_F": 2.1
        }
    }
] * 200   # 200 historical rows

@app.get("/api/v1/trend/history")
def get_history():
    monitor_id = request.args.get("monitorId")
    months = request.args.get("months")

    if not monitor_id:
        return jsonify({"success": False, "error": "Missing monitorId"}), 400

    print(f"API called for monitor {monitor_id} months={months}")

    return jsonify({
        "success": True,
        "records": DUMMY_DATA   # Always return same data
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
