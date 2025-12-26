#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ------------ CONFIG ------------
const char* ssid     = "Redmi";
const char* password = "123456789";
// Replace with your actual Cloud Server URL (e.g., http://your-app.herokuapp.com/data)
//const char* serverUrl = "http://172.31.34.161/data"; 
const char* serverUrl ="https://capstone-mhm6.onrender.com/data";
const int PIN_MCP_CS   = 5;
const int PIN_MCP_CLK  = 18;
const int PIN_MCP_MISO = 19;
const int PIN_MCP_MOSI = 23;
const int PIN_SPEAKER  = 25;

const float START_FREQ_HZ = 150.0f;
const float MIN_FREQ_HZ   = 2.0f;
const unsigned long SWEEP_DURATION_US = 5000000UL;

// Buffer to store data points during the sweep
const int MAX_SAMPLES = 500; 
struct Reading {
  uint32_t t;
  uint16_t v;
};
Reading buffer[MAX_SAMPLES];
int count = 0;

// ------------ MCP3008 READ ------------
uint16_t readMCP3008(uint8_t channel) {
  digitalWrite(PIN_MCP_CS, LOW);
  SPI.transfer(0x01);
  uint8_t hi = SPI.transfer(0x80 | (channel << 4));
  uint8_t lo = SPI.transfer(0x00);
  digitalWrite(PIN_MCP_CS, HIGH);
  return ((hi & 0x03) << 8) | lo;
}

// ------------ SEND DATA TO CLOUD ------------
void uploadData() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  // Create JSON document
  DynamicJsonDocument doc(16384); 
  JsonArray readings = doc.createNestedArray("readings");

  for (int i = 0; i < count; i++) {
    JsonObject obj = readings.createNestedObject();
    obj["t"] = buffer[i].t;
    obj["v"] = buffer[i].v;
  }

  String jsonExport;
  serializeJson(doc, jsonExport);

  Serial.println("Uploading to server...");
  int httpResponseCode = http.POST(jsonExport);
  
  if (httpResponseCode > 0) {
    Serial.print("Server Response: ");
    Serial.println(httpResponseCode);
  } else {
    Serial.print("Error: ");
    Serial.println(http.errorToString(httpResponseCode).c_str());
  }
  http.end();
}

// ------------ SWEEP LOGIC ------------
void runSweep() {
  unsigned long t0 = micros();
  count = 0;

  while (true) {
    unsigned long elapsed = micros() - t0;
    if (elapsed >= SWEEP_DURATION_US) break;

    float frac = (float)elapsed / (float)SWEEP_DURATION_US;
    float f = START_FREQ_HZ * (1.0f - frac);
    if (f < MIN_FREQ_HZ) f = MIN_FREQ_HZ;

    unsigned long periodUs = 1000000.0f / f;
    unsigned long halfPeriod = periodUs / 2;

    // PULSE (Audio)
    digitalWrite(PIN_SPEAKER, HIGH);
    delayMicroseconds(halfPeriod);
    digitalWrite(PIN_SPEAKER, LOW);
    delayMicroseconds(halfPeriod);

    // CAPTURE (ADC) - Store in RAM
    if (count < MAX_SAMPLES) {
      buffer[count].t = elapsed / 1000; // time in ms
      buffer[count].v = readMCP3008(0);
      count++;
    }
  }
  digitalWrite(PIN_SPEAKER, LOW);
  Serial.println("Sweep complete.");
  uploadData();
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_SPEAKER, OUTPUT);
  pinMode(PIN_MCP_CS, OUTPUT);
  digitalWrite(PIN_MCP_CS, HIGH);
  SPI.begin(PIN_MCP_CLK, PIN_MCP_MISO, PIN_MCP_MOSI);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi Connected");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S') runSweep();
  }
}