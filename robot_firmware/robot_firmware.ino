#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ---------- Local Wi-Fi Network Credentials ----------
// The ESP32 now JOINS your home/router Wi-Fi instead of hosting its own
// access point. Set these to your actual network name and password.
const char* WIFI_SSID     = "WE-5G";
const char* WIFI_PASSWORD = "159@Mahmoud147";

WebServer server(80);

// Initialize PCA9685 at default I2C address (0x40)
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// Microsecond limits for standard 0-180 degree translation mapping
#define USMIN  750
#define USMAX  2400
#define SERVO_FREQ 50

// ---------- DC Motor Pins ----------
#define FL_IN1 27
#define FL_IN2 14
#define FL_ENA 26

#define FR_IN1 33
#define FR_IN2 25
#define FR_ENB 32

#define RL_IN1 18
#define RL_IN2 19
#define RL_ENA 23

#define RR_IN1 12
#define RR_IN2 15
#define RR_ENB 13

// ---------- Speed Control ----------
// 0-255 PWM duty applied to all enable pins. Adjustable via /speed?value=
int currentSpeed = 255;

// ---------- Servo Constraint Configuration ----------
// NUM_SERVOS = 8 original + 2 new ones (channels 8 and 9 on the PCA9685).
// Rename/re-tune the new channels below to match whatever you actually
// mount there (gripper, waist rotation, extra head axis, etc).
#define NUM_SERVOS 10

struct ServoConfig {
  int minAngle;
  int maxAngle;
  int startAngle;
};

ServoConfig servoLimits[NUM_SERVOS] = {
  {45,   135,  90},  // Servo 0: head horizontal
  {60,   125,  90},  // Servo 1: arm rotate right
  {60,   125,  125}, // Servo 2: arm right above
  {60,   125,  90},  // Servo 3: arm rotate left
  {60,   125,  60},  // Servo 4: arm left above
  {45,   135,  90},  // Servo 5: arm left under
  {45,   135,  90},  // Servo 6: arm right under
  {60,   115,  80},  // Servo 7: head vertical
  {70,    115,  90},  // Servo 8: gripper right
  {70,    115,  90}   // Servo 9: gripper left
};

// ---------- Motor Control Logic ----------
void enableAllMotors() {
  analogWrite(FL_ENA, currentSpeed);
  analogWrite(FR_ENB, currentSpeed);
  analogWrite(RL_ENA, currentSpeed);
  analogWrite(RR_ENB, currentSpeed);
}

void stopMotors() {
  digitalWrite(FL_ENA, LOW);   digitalWrite(FR_ENB, LOW);
  digitalWrite(RL_ENA, LOW);   digitalWrite(RR_ENB, LOW);
  digitalWrite(FL_IN1, LOW);  digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, LOW);  digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, LOW);  digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, LOW);  digitalWrite(RR_IN2, LOW);
}

void forward() {
  enableAllMotors();
  digitalWrite(FL_IN1, HIGH); digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, HIGH); digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, HIGH); digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, HIGH); digitalWrite(RR_IN2, LOW);
}

void backward() {
  enableAllMotors();
  digitalWrite(FL_IN1, LOW); digitalWrite(FL_IN2, HIGH);
  digitalWrite(FR_IN1, LOW); digitalWrite(FR_IN2, HIGH);
  digitalWrite(RL_IN1, LOW); digitalWrite(RL_IN2, HIGH);
  digitalWrite(RR_IN1, LOW); digitalWrite(RR_IN2, HIGH);
}

// Sideways strafe (mecanum), no rotation
void strafeLeft() {
  enableAllMotors();
  digitalWrite(FL_IN1, LOW);  digitalWrite(FL_IN2, HIGH);
  digitalWrite(FR_IN1, HIGH); digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, HIGH); digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, LOW);  digitalWrite(RR_IN2, HIGH);
}

void strafeRight() {
  enableAllMotors();
  digitalWrite(FL_IN1, HIGH); digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, LOW);  digitalWrite(FR_IN2, HIGH);
  digitalWrite(RL_IN1, LOW);  digitalWrite(RL_IN2, HIGH);
  digitalWrite(RR_IN1, HIGH); digitalWrite(RR_IN2, LOW);
}

void rotateLeft() {
  enableAllMotors();
  digitalWrite(FL_IN1, LOW);  digitalWrite(FL_IN2, HIGH);
  digitalWrite(FR_IN1, HIGH); digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, LOW);  digitalWrite(RL_IN2, HIGH);
  digitalWrite(RR_IN1, HIGH); digitalWrite(RR_IN2, LOW);
}

void rotateRight() {
  enableAllMotors();
  digitalWrite(FL_IN1, HIGH); digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, LOW);  digitalWrite(FR_IN2, HIGH);
  digitalWrite(RL_IN1, HIGH); digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, LOW);  digitalWrite(RR_IN2, HIGH);
}

// ---------- Diagonal (true mecanum) moves ----------
// Only the two wheels on the active diagonal spin; the other two stay idle.
void diagForwardRight() {
  enableAllMotors();
  digitalWrite(FL_IN1, HIGH); digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, LOW);  digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, LOW);  digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, HIGH); digitalWrite(RR_IN2, LOW);
}

void diagForwardLeft() {
  enableAllMotors();
  digitalWrite(FL_IN1, LOW);  digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, HIGH); digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, HIGH); digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, LOW);  digitalWrite(RR_IN2, LOW);
}

void diagBackwardRight() {
  enableAllMotors();
  digitalWrite(FL_IN1, LOW);  digitalWrite(FL_IN2, LOW);
  digitalWrite(FR_IN1, LOW);  digitalWrite(FR_IN2, HIGH);
  digitalWrite(RL_IN1, LOW);  digitalWrite(RL_IN2, HIGH);
  digitalWrite(RR_IN1, LOW);  digitalWrite(RR_IN2, LOW);
}

void diagBackwardLeft() {
  enableAllMotors();
  digitalWrite(FL_IN1, LOW);  digitalWrite(FL_IN2, HIGH);
  digitalWrite(FR_IN1, LOW);  digitalWrite(FR_IN2, LOW);
  digitalWrite(RL_IN1, LOW);  digitalWrite(RL_IN2, LOW);
  digitalWrite(RR_IN1, LOW);  digitalWrite(RR_IN2, HIGH);
}

// ---------- PCA9685 Servo Helper with Constraints ----------
void moveServo(int channel, int angle) {
  if (channel < 0 || channel >= NUM_SERVOS) return;
  int constrainedAngle = constrain(angle, servoLimits[channel].minAngle, servoLimits[channel].maxAngle);
  int pulseLen = map(constrainedAngle, 0, 180, USMIN, USMAX);
  pwm.writeMicroseconds(channel, pulseLen);
}

// ---------- HTML Webpage Interface ----------
String page = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot & Servo Interface</title>
<style>
body { font-family: Arial, sans-serif; text-align: center; background: #222; color: white; padding: 20px 0; }
button {
  width: 110px; height: 60px; font-size: 16px; margin: 8px; border-radius: 10px;
  border: none; background-color: #444; color: white; font-weight: bold; cursor: pointer;
  box-shadow: 0 4px #111; transition: all 0.1s ease;
}
button:active { background-color: #666; box-shadow: 0 1px #111; transform: translateY(3px); }
.slider-container { width: 80%; max-width: 400px; margin: 15px auto; background: #333; padding: 10px; border-radius: 8px; }
.slider { width: 100%; }
.row { margin: 5px 0; }
.section { margin-top: 25px; border-top: 1px solid #555; padding-top: 15px; }
</style>
<script>
function sendCommand(path) {
  fetch(path).catch(err => console.error('Transmission error:', err));
}
function updateServo(num, val) {
  document.getElementById("val" + num).innerText = val + "\u00b0";
  fetch('/servo?num=' + num + '&angle=' + val).catch(err => console.error(err));
}
</script>
</head>
<body>

<h2>Chassis Control</h2>
<div class="row">
  <button onclick="sendCommand('/diagfl')">&#8598; FwdL</button>
  <button onclick="sendCommand('/forward')">Forward</button>
  <button onclick="sendCommand('/diagfr')">FwdR &#8599;</button>
</div>
<div class="row">
  <button onclick="sendCommand('/left')">Strafe L</button>
  <button onclick="sendCommand('/stop')" style="background-color: #a33;">Stop</button>
  <button onclick="sendCommand('/right')">Strafe R</button>
</div>
<div class="row">
  <button onclick="sendCommand('/diagbl')">&#8601; BackL</button>
  <button onclick="sendCommand('/backward')">Backward</button>
  <button onclick="sendCommand('/diagbr')">BackR &#8600;</button>
</div>
<div class="row">
  <button onclick="sendCommand('/rotateleft')">Rotate L</button>
  <button onclick="sendCommand('/rotateright')">Rotate R</button>
</div>

<div class="section">
  <h2>Servo Configuration (Channels 0 - 9)</h2>

  )rawliteral";

String getSliderHtml(int num) {
  String html = "<div class='slider-container'>";
  html += "<label>Servo " + String(num) + " (Limits: " + String(servoLimits[num].minAngle) + "\u00b0-" + String(servoLimits[num].maxAngle) + "\u00b0): <span id='val" + String(num) + "'>" + String(servoLimits[num].startAngle) + "\u00b0</span></label>";
  html += "<input type='range' min='0' max='180' value='" + String(servoLimits[num].startAngle) + "' class='slider' oninput='updateServo(" + String(num) + ", this.value)'>";
  html += "</div>";
  return html;
}

// ---------- Initialization and Setup ----------
void setup() {
  Serial.begin(115200);

  pinMode(FL_IN1, OUTPUT); pinMode(FL_IN2, OUTPUT); pinMode(FL_ENA, OUTPUT);
  pinMode(FR_IN1, OUTPUT); pinMode(FR_IN2, OUTPUT); pinMode(FR_ENB, OUTPUT);
  pinMode(RL_IN1, OUTPUT); pinMode(RL_IN2, OUTPUT); pinMode(RL_ENA, OUTPUT);
  pinMode(RR_IN1, OUTPUT); pinMode(RR_IN2, OUTPUT); pinMode(RR_ENB, OUTPUT);
  stopMotors();

  Wire.begin(21, 22);

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  for (int i = 0; i < NUM_SERVOS; i++) {
    moveServo(i, servoLimits[i].startAngle);
  }

  // ----- Connect to your local Wi-Fi network (router) -----
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  unsigned long startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
    // After 20s, restart the attempt rather than hanging forever
    if (millis() - startAttempt > 20000) {
      Serial.println("\nRetrying Wi-Fi connection...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      startAttempt = millis();
    }
  }
  Serial.println("\nConnected!");
  Serial.print("Robot IP address: ");
  Serial.println(WiFi.localIP());  // <-- use this IP in the Python Robot() client

  server.on("/", []() {
    String fullPage = page;
    for (int i = 0; i < NUM_SERVOS; i++) {
      fullPage += getSliderHtml(i);
    }
    fullPage += "</div></body></html>";
    server.sendHeader("Connection", "close");
    server.send(200, "text/html", fullPage);
  });

  // ----- Chassis routes -----
  server.on("/forward",     []() { forward();          server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/backward",    []() { backward();         server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/left",        []() { strafeLeft();       server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/right",       []() { strafeRight();      server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/rotateleft",  []() { rotateLeft();       server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/rotateright", []() { rotateRight();      server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/diagfl",      []() { diagForwardLeft();  server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/diagfr",      []() { diagForwardRight(); server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/diagbl",      []() { diagBackwardLeft(); server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/diagbr",      []() { diagBackwardRight();server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });
  server.on("/stop",        []() { stopMotors();       server.sendHeader("Connection", "close"); server.send(200, "text/plain", "OK"); });

  // Speed control: /speed?value=0-255
  server.on("/speed", []() {
    if (server.hasArg("value")) {
      currentSpeed = constrain(server.arg("value").toInt(), 0, 255);
    }
    server.sendHeader("Connection", "close");
    server.send(200, "text/plain", "OK");
  });

  // Single servo: /servo?num=0-9&angle=0-180 (range clamped per-channel below)
  server.on("/servo", []() {
    if (server.hasArg("num") && server.hasArg("angle")) {
      int servoNum = server.arg("num").toInt();
      int servoAngle = server.arg("angle").toInt();
      moveServo(servoNum, servoAngle);
    }
    server.sendHeader("Connection", "close");
    server.send(200, "text/plain", "OK");
  });

  // Atomic multi-servo pose: /pose?s0=90&s3=45&s9=80  (any subset of s0..s9)
  server.on("/pose", []() {
    for (int i = 0; i < NUM_SERVOS; i++) {
      String key = "s" + String(i);
      if (server.hasArg(key)) {
        moveServo(i, server.arg(key).toInt());
      }
    }
    server.sendHeader("Connection", "close");
    server.send(200, "text/plain", "OK");
  });

  // Capability/status report for clients
  server.on("/status", []() {
    String json = "{\"speed\":" + String(currentSpeed) + ",\"servos\":[";
    for (int i = 0; i < NUM_SERVOS; i++) {
      json += "{\"channel\":" + String(i) +
              ",\"min\":" + String(servoLimits[i].minAngle) +
              ",\"max\":" + String(servoLimits[i].maxAngle) +
              ",\"start\":" + String(servoLimits[i].startAngle) + "}";
      if (i < NUM_SERVOS - 1) json += ",";
    }
    json += "]}";
    server.sendHeader("Connection", "close");
    server.send(200, "application/json", json);
  });

  server.begin();
}

void loop() {
  server.handleClient();
}
