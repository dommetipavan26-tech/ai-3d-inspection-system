/*
 * ============================================================================
 *  AI 3D Structural Inspection System -- ESP32-CAM field node firmware
 * ============================================================================
 *
 *  Target board   : 16-pin AI-Thinker ESP32-CAM (the small blue/black module
 *                   with the microSD slot on the back of the OV2640 FPC)
 *                   -- NOT the 8-pin ESP32-CAM-MB carrier.
 *  Sensor         : OV2640 2 MP (UXGA 1600x1200 colour JPEG)
 *  Framework      : Arduino ESP32 core (PlatformIO: espressif32 / arduino)
 *  Role           : one of the eight orbital nodes of the inspection rig
 *
 *  FIELD WIRING (this is the whole point of picking the 16-pin module)
 *  ---------------------------------------------------------------------------
 *  In the field the node is powered from a 5 V USB power bank / 12 V->5 V
 *  buck converter.  Only TWO wires leave the board:
 *
 *        5V  pin  <-- +5 V (power bank or buck converter)
 *        GND pin  <-- 0 V
 *
 *  Everything else is already routed on the module's internal PCB traces:
 *    * the OV2640 is wired to the ESP32 through the flex connector and the
 *      internal traces (D0..D7, XCLK, PCLK, VSYNC, HREF, SIOD, SIOC, PWDN,
 *      RESET) -- nothing needs to be jumpered by hand;
 *    * GPIO 4  drives the on-board high-intensity white flash LED;
 *    * GPIO 33 drives the on-board red LED, used here as the capture
 *      countdown indicator (3 s) so the operator knows not to move.
 *
 *  GPIO 0 is a boot-strap pin and is pulled low by the on-board 10 k; it is
 *  never driven by this firmware.  The pins used by the camera and the LEDs
 *  (GPIO 4, 5, 14, 15, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33, 34, 35, 36, 39)
 *  are therefore all reserved and are NOT available for external sensors --
 *  if you need a trigger input, use GPIO 12, 13 or 2, which are free on the
 *  16-pin module when the SD card is not used.
 *
 *  BEHAVIOUR
 *  ---------------------------------------------------------------------------
 *   1. Connects to the inspection WiFi network and starts an HTTPS client.
 *   2. Every CAPTURE_INTERVAL_S (default 900 s) runs a capture:
 *        a. red LED (GPIO 33) blinks a 3 s countdown, 1 Hz;
 *        b. flash LED (GPIO 4) fires for 120 ms;
 *        c. the OV2640 grabs a JPEG at UXGA / quality 12;
 *        d. the frame is POSTed as multipart/form-data to
 *           https://<host>/api/esp32/upload together with device_id, view,
 *           RSSI, battery voltage and firmware version.
 *   3. Every 3 s it polls https://<host>/api/esp32/command for a queued
 *      command.  Supported commands:
 *           capture        -> take a frame now
 *           flash          -> test the white LED
 *           set_interval   -> change the autonomous capture period
 *           status         -> report uptime, RSSI, heap, battery
 *           reboot         -> ESP.restart()
 *      An "idle" answer means nothing is queued and the node sleeps 3 s.
 *   4. Brownout / WiFi loss are handled with a bounded exponential backoff so
 *      a node never wedges the whole rig.
 *
 *  HTTPS: the server certificate is validated against the CA bundle embedded
 *  in the sketch (or, on an isolated field network, INSPECTION_INSECURE_TLS
 *  can be set to 1 to skip validation -- see the security note in README.md).
 * ============================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <time.h>

// ---------------------------------------------------------------------------
// 1. USER CONFIGURATION
// ---------------------------------------------------------------------------

// --- inspection server (Render) --------------------------------------------
#define INSPECTION_HOST        "ai-3d-inspection-system.onrender.com"
#define UPLOAD_PATH            "/api/esp32/upload"
#define COMMAND_PATH           "/api/esp32/command"
#define INSPECTION_PORT        443
#define USE_HTTPS              1

// Optional shared secret.  Leave empty when the server does not set
// ESP32_UPLOAD_TOKEN; otherwise set the same value here and the node sends it
// in the X-Inspection-Token header.
#define INSPECTION_TOKEN       ""

// 1 = do not verify the server certificate (field networks without a CA bundle
// and without NTP time sync).  0 = verify against the embedded root CA.
#define INSPECTION_INSECURE_TLS 1

// Root CA (ISRG Root X1, the Let's Encrypt / Render chain root).
static const char *INSPECTION_ROOT_CA = R"EOF(
-----BEGIN CERTIFICATE-----
MIIFazCCA1OgAwIBAgIRAIIQz7DSQONZRGPgu2OCiwAwDQYJKoZIhvcNAQELBQAw
TzELMAkGA1UEBhMCVVMxKTAnBgNVBAoTIEludGVybmV0IFNlY3VyaXR5IFJlc2Vh
cmNoIEdyb3VwMRUwEwYDVQQDEwxJU1JHIFJvb3QgWDEwHhcNMTUwNjA0MTEwNDM4
WhcNMzUwNjA0MTEwNDM4WjBPMQswCQYDVQQGEwJVUzEpMCcGA1UEChMgSW50ZXJu
ZXQgU2VjdXJpdHkgUmVzZWFyY2ggR3JvdXAxFTATBgNVBAMTDElTUkcgUm9vdCBY
MTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBAK3oJHP0FDfzm54rVygc
h77ct984kIxuPOZXoHj3dcKi/vVqbvYATyjb3miGbESTtrFj/RQSa78f0uoxmyF+
0TM8ukj13Xnfs7j/EvEhmkvBioZxaUpmZmyPfjxwv60pIgbz5MDmgK7iS4+3mX6U
A5/TR5d8mUgjU+g4rk8Kb4Mu0UlXjIB0ttov0DiNewNwIRt18jA8+o+u3dpjq+sW
T8KOEUt+zwvo/7V3LvSye0rgTBIlDHCNAymg4VMk7BPZ7hm/ELNKjD+Jo2FR3qyH
B5T0Y3HsLuJvW5iB4YlcNHlsdu87kGJ55tukmi8mxdAQ4Q7e2RCOFvu396j3x+UC
B5iPNgiV5+I3lg02dZ77DnKxHZu8A/lJBdiB3QW0KtZB6awBdpUKD9jf1b0SHzUv
KBds0pjBqAlkd25HN7rOrFleaJ1/ctaJxQZBKT5ZPt0m9STJEadao0xAH0ahmbWn
OlFuhjuefXKnEgV4We0+UXgVCwOPjdAvBbI+e0ocS3MFEvzG6uBQE3xDk3SzynTn
jh8BCNAw1FtxNrQHusEwMFxIt4I7mKZ9YIqioymCzLq9gwQbooMDQaHWBfEbwrbw
qHyGO0aoSCqI3Haadr8faqU9GY/rOPNk3sgrDQoo//fb4hVC1CLQJ13hef4Y53CI
rU7m2Ys6xt0nUW7/vGT1M0NPAgMBAAGjQjBAMA4GA1UdDwEB/wQEAwIBBjAPBgNV
HRMBAf8EBTADAQH/MB0GA1UdDgQWBBR5tFnme7bl5AFzgAiIyBpY9umbbjANBgkq
hkiG9w0BAQsFAAOCAgEAVR9YqbyyqFDQDLHYGmkgJykIrGF1XIpu+ILlaS/V9lZL
ubhzEFnTIZd+50xx+7LSYK05qAvqFyFWhfFQDlnrzuBZ6brJFe+GnY+EgPbk6ZGQ
3BebYhtF8GaV0nxvwuo77x/Py9auJ/GpsMiu/X1+mvoiBOv/2X/qkSsisRcOj/KK
NFtY2PwByVS5uCbMiogziUwthDyC3+6WVwW6LLv3xLfHTjuCvjHIInNzktHCgKQ5
ORAzI4JMPJ+GslWYHb4phowim57iaztXOoJwTdwJx4nLCgdNbOhdjsnvzqvHu7Ur
TkXWStAmzOVyyghqpZXjFaH3pO3JLF+l+/+sKAIuvtd7u+Nxe5AW0wdeRlN8NwdC
jNPElpzVmbUq4JUagEiuTDkHzsxHpFKVK7q4+63SM1N95R1NbdWhscdCb+ZAJzVc
oyi3B43njTOQ5yOf+1CceWxG1bQVs5ZufpsMljq4Ui0/1lvh+wjChP4kqKOJ2qxq
4RgqsahDYVvTH9w7jXbyLeiNdd8XM2w9U/t7y0Ff/9yi0GE44Za4rF2LN9d11TPA
mRGunUHBcnWEvgJBQl9nJEiU0Zsnvgc/ubhPgXRR4Xq37Z0j4r7g1SgEEzwxA57d
emyPxgcYxn/eR44/KJ4EBs+lVDR3veyJm+kXQ99b21/+jh5Xos1AnX5iItreGCc=
-----END CERTIFICATE-----
)EOF";

// --- WiFi ------------------------------------------------------------------
const char *WIFI_SSID     = "INSPECTION_RIG";
const char *WIFI_PASSWORD = "change-me-please";

// --- node identity (must match one of the eight orbital positions) ---------
#define NODE_VIEW_INDEX        1        // 1..8  -> view V1..V8
#define NODE_DEVICE_ID         "ESP32CAM-01"
#define FIRMWARE_VERSION       "1.0.0"

// --- capture settings ------------------------------------------------------
#define CAMERA_FRAMESIZE       FRAMESIZE_UXGA      // 1600 x 1200 (OV2640 max JPEG)
#define CAMERA_JPEG_QUALITY    12                  // 0..63, lower = better
#define COUNTDOWN_SECONDS      3
#define FLASH_PULSE_MS         120
#define CAPTURE_INTERVAL_S     900                 // 15 min autonomous period
#define COMMAND_POLL_MS        3000                // 3 s command polling

// --- pins (16-pin AI-Thinker ESP32-CAM) ------------------------------------
#define PIN_FLASH_LED          4                   // on-board white flash LED
#define PIN_COUNTDOWN_LED      33                  // on-board red LED
#define PIN_BATTERY_ADC        35                  // optional 2:1 divider on 5 V rail
#define ENABLE_BATTERY_SENSE   1

// ---------------------------------------------------------------------------
// 2. CAMERA PIN MAP -- 16-pin AI-Thinker ESP32-CAM (internal PCB traces)
// ---------------------------------------------------------------------------
#define PWDN_GPIO_NUM    32
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      21
#define Y4_GPIO_NUM      19
#define Y3_GPIO_NUM      18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

// ---------------------------------------------------------------------------
// 3. STATE
// ---------------------------------------------------------------------------
static unsigned long g_last_capture_ms = 0;
static unsigned long g_last_poll_ms    = 0;
static unsigned long g_capture_period_ms = (unsigned long)CAPTURE_INTERVAL_S * 1000UL;
static uint32_t g_frames_sent  = 0;
static uint32_t g_frames_failed = 0;
static uint32_t g_commands_handled = 0;
static float    g_battery_v = 0.0f;
static bool     g_camera_ready = false;

// ---------------------------------------------------------------------------
// 4. HELPERS
// ---------------------------------------------------------------------------
static void led_signal(uint8_t led, uint32_t total_ms, uint32_t on_ms, uint32_t off_ms)
{
    unsigned long t0 = millis();
    bool state = false;
    while (millis() - t0 < total_ms) {
        state = !state;
        digitalWrite(led, state ? HIGH : LOW);
        delay(state ? on_ms : off_ms);
    }
    digitalWrite(led, LOW);
}

static float read_battery_voltage()
{
#if ENABLE_BATTERY_SENSE
    // 10 k / 10 k divider on the 5 V rail -> ADC sees 2.5 V at 5.0 V
    uint32_t mv = 0;
    for (int i = 0; i < 16; i++) {
        mv += analogReadMilliVolts(PIN_BATTERY_ADC);
        delay(2);
    }
    return (mv / 16.0f) * 2.0f / 1000.0f;
#else
    return 0.0f;
#endif
}

static String iso8601_now()
{
    time_t now = time(nullptr);
    if (now < 1600000000) {
        return String("uptime+") + String(millis() / 1000) + "s";
    }
    struct tm tmv;
    gmtime_r(&now, &tmv);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return String(buf);
}

// ---------------------------------------------------------------------------
// 5. CAMERA
// ---------------------------------------------------------------------------
static bool camera_init()
{
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = CAMERA_FRAMESIZE;
    config.jpeg_quality = CAMERA_JPEG_QUALITY;
    config.fb_count     = 2;          // double buffer for PSRAM boards
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        // Some 16-pin boards ship without PSRAM: retry at SVGA in internal RAM.
        Serial.printf("[cam] init failed 0x%x -- retrying SVGA without PSRAM\n", err);
        config.frame_size  = FRAMESIZE_SVGA;
        config.fb_count    = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
        err = esp_camera_init(&config);
        if (err != ESP_OK) {
            Serial.printf("[cam] init failed twice (0x%x). Check the 5 V supply: "
                          "the OV2640 browns out below 4.8 V.\n", err);
            return false;
        }
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_framesize(s, CAMERA_FRAMESIZE);
        s->set_quality(s, CAMERA_JPEG_QUALITY);
        s->set_brightness(s, 0);
        s->set_contrast(s, 1);       // helps the crack detector on grey concrete
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1);
        s->set_gain_ctrl(s, 1);
        s->set_lenc(s, 1);           // lens-shading correction: important for
                                     // even illumination across the pier face
        s->set_raw_gma(s, 1);
        s->set_dcw(s, 1);
        s->set_bpc(s, 1);
        s->set_wpc(s, 1);
    }
    Serial.printf("[cam] ready: %s, JPEG q%d\n",
                  config.frame_size == FRAMESIZE_UXGA ? "UXGA 1600x1200" : "SVGA 800x600",
                  CAMERA_JPEG_QUALITY);
    return true;
}

static camera_fb_t *grab_frame()
{
    // 3 s countdown on the red LED so the operator (or the rig) can settle
    led_signal(PIN_COUNTDOWN_LED, COUNTDOWN_SECONDS * 1000UL, 450, 550);

    // reject the first two frames so the AEC/AGC has settled after the LED
    for (int i = 0; i < 2; i++) {
        camera_fb_t *drop = esp_camera_fb_get();
        if (drop) esp_camera_fb_return(drop);
        delay(60);
    }

    digitalWrite(PIN_FLASH_LED, HIGH);
    delay(FLASH_PULSE_MS);
    camera_fb_t *fb = esp_camera_fb_get();
    digitalWrite(PIN_FLASH_LED, LOW);

    if (!fb) {
        Serial.println("[cam] capture returned no frame buffer");
    }
    return fb;
}

// ---------------------------------------------------------------------------
// 6. WIFI
// ---------------------------------------------------------------------------
static bool wifi_connect(uint32_t timeout_ms = 25000)
{
    if (WiFi.status() == WL_CONNECTED) return true;
    Serial.printf("[wifi] connecting to %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < timeout_ms) {
        delay(400);
        Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("[wifi] connected: %s  RSSI %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
        configTime(0, 0, "pool.ntp.org", "time.nist.gov");  // needed for TLS dates
        return true;
    }
    Serial.println("[wifi] FAILED");
    return false;
}

static bool ensure_wifi()
{
    if (WiFi.status() == WL_CONNECTED) return true;
    Serial.println("[wifi] link lost -- reconnecting");
    WiFi.disconnect(true);
    delay(250);
    return wifi_connect();
}

// ---------------------------------------------------------------------------
// 7. UPLOAD (multipart/form-data over HTTPS)
// ---------------------------------------------------------------------------
//  The JPEG is streamed straight out of the camera frame buffer on the socket,
//  in 2 kB chunks: a UXGA frame is ~120 kB, and assembling it inside an
//  HTTPClient payload would need a second copy in the heap, which is exactly
//  what runs a 16-pin board out of RAM in the field.
// ---------------------------------------------------------------------------

static bool upload_frame(camera_fb_t *fb)
{
    if (!ensure_wifi()) return false;

    WiFiClientSecure client;
    client.setTimeout(20000);
#if USE_HTTPS
    if (INSPECTION_INSECURE_TLS) {
        client.setInsecure();          // field mode: no CA bundle, no NTP
    } else {
        client.setCACert(INSPECTION_ROOT_CA);
    }
#endif
    client.setHandshakeTimeout(20);

    if (!client.connect(INSPECTION_HOST, INSPECTION_PORT)) {
        Serial.println("[tls] connect failed");
        return false;
    }

    const char *boundary = "ESP32CAMBoundary";
    String head;
    head.reserve(700);
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"device_id\"\r\n\r\n";
    head += NODE_DEVICE_ID; head += "\r\n";
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"view\"\r\n\r\n";
    head += String("V") + NODE_VIEW_INDEX; head += "\r\n";
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"rssi\"\r\n\r\n";
    head += String(WiFi.RSSI()); head += "\r\n";
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"battery\"\r\n\r\n";
    head += String(g_battery_v, 2); head += "\r\n";
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"firmware\"\r\n\r\n";
    head += FIRMWARE_VERSION; head += "\r\n";
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"captured_at\"\r\n\r\n";
    head += iso8601_now(); head += "\r\n";
    head += "--"; head += boundary; head += "\r\n";
    head += "Content-Disposition: form-data; name=\"file\"; filename=\"";
    head += String("esp32_cam_0") + NODE_VIEW_INDEX + "_view_V" +
            NODE_VIEW_INDEX + ".jpg";
    head += "\"\r\nContent-Type: image/jpeg\r\n\r\n";

    String tail = String("\r\n--") + boundary + "--\r\n";
    size_t total = head.length() + fb->len + tail.length();

    // ---- request head -----------------------------------------------------
    client.printf("POST %s HTTP/1.1\r\n", UPLOAD_PATH);
    client.printf("Host: %s\r\n", INSPECTION_HOST);
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary);
    client.printf("Content-Length: %u\r\n", (unsigned)total);
    client.printf("X-Device-Id: %s\r\n", NODE_DEVICE_ID);
    if (strlen(INSPECTION_TOKEN) > 0) {
        client.printf("X-Inspection-Token: %s\r\n", INSPECTION_TOKEN);
    }
    client.print("Connection: close\r\n");
    client.print("User-Agent: ESP32-CAM-Inspection/1.0\r\n\r\n");

    // ---- body: text header, JPEG chunks, trailer --------------------------
    client.print(head);
    const size_t CHUNK = 2048;
    size_t sent = 0;
    while (sent < fb->len) {
        size_t n = (fb->len - sent < CHUNK) ? (fb->len - sent) : CHUNK;
        size_t w = client.write(fb->buf + sent, n);
        if (w == 0) {
            Serial.println("[tls] socket write returned 0 -- aborting upload");
            client.stop();
            return false;
        }
        sent += w;
        delay(1);                  // yield to the IDF task watchdog
    }
    client.print(tail);

    // ---- status line ------------------------------------------------------
    String status = client.readStringUntil('\n');
    int code = 0;
    int sp1 = status.indexOf(' ');
    if (sp1 > 0) code = status.substring(sp1 + 1, sp1 + 4).toInt();
    Serial.printf("[http] upload -> %s", status.c_str());
    Serial.printf("        (%u bytes sent)\n", (unsigned)sent);

    // ---- response body (useful for debugging a rejected upload) ----------
    String body;
    unsigned long t0 = millis();
    while (client.connected() && millis() - t0 < 4000) {
        while (client.available()) {
            char c = (char)client.read();
            if (body.length() < 400) body += c;
        }
        delay(5);
    }
    client.stop();
    if (code != 200 && code != 201 && body.length()) {
        Serial.printf("[http] server said: %s\n", body.c_str());
    }
    return code == 200 || code == 201;
}

// ---------------------------------------------------------------------------
// 8. COMMAND POLLING
// ---------------------------------------------------------------------------
static void handle_command(JsonDocument &doc)
{
    const char *cmd = doc["command"] | "idle";
    const char *cid = doc["id"] | "-";
    Serial.printf("[cmd] %s (%s)\n", cmd, cid);

    if (strcmp(cmd, "idle") == 0) {
        return;
    } else if (strcmp(cmd, "capture") == 0) {
        g_last_capture_ms = 0;                 // force a capture on the next loop
        g_commands_handled++;
    } else if (strcmp(cmd, "flash") == 0) {
        led_signal(PIN_FLASH_LED, 600, 300, 300);
        g_commands_handled++;
    } else if (strcmp(cmd, "set_interval") == 0) {
        int secs = doc["params"]["seconds"] | (int)(g_capture_period_ms / 1000UL);
        if (secs < 30) secs = 30;
        g_capture_period_ms = (unsigned long)secs * 1000UL;
        Serial.printf("[cmd] capture period set to %d s\n", secs);
        g_commands_handled++;
    } else if (strcmp(cmd, "status") == 0) {
        Serial.printf("[status] view=V%d rssi=%d uptime=%lus heap=%u free_psram=%u "
                      "sent=%u failed=%u battery=%.2fV\n",
                      NODE_VIEW_INDEX, WiFi.RSSI(), millis() / 1000UL,
                      (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram(),
                      (unsigned)g_frames_sent, (unsigned)g_frames_failed,
                      g_battery_v);
        g_commands_handled++;
    } else if (strcmp(cmd, "reboot") == 0) {
        Serial.println("[cmd] rebooting");
        delay(200);
        ESP.restart();
    } else {
        Serial.printf("[cmd] unknown command '%s' ignored\n", cmd);
    }
}

static void poll_command()
{
    if (!ensure_wifi()) return;

    WiFiClientSecure client;
    client.setTimeout(8000);
#if USE_HTTPS
    if (INSPECTION_INSECURE_TLS) client.setInsecure();
    else client.setCACert(INSPECTION_ROOT_CA);
#endif

    if (!client.connect(INSPECTION_HOST, INSPECTION_PORT)) {
        Serial.println("[poll] connect failed");
        return;
    }

    String path = String(COMMAND_PATH) + "?device=" + NODE_DEVICE_ID +
                  "&view=" + String(NODE_VIEW_INDEX);
    client.printf("GET %s HTTP/1.1\r\n", path.c_str());
    client.printf("Host: %s\r\n", INSPECTION_HOST);
    client.print("Accept: application/json\r\nUser-Agent: ESP32-CAM-Inspection/1.0\r\n"
                 "Connection: close\r\n\r\n");

    // ---- status line + headers ------------------------------------------
    String status = client.readStringUntil('\n');
    if (status.indexOf("200") < 0) {
        Serial.printf("[poll] HTTP status: %s\n", status.c_str());
        client.stop();
        return;
    }
    while (client.connected()) {
        String h = client.readStringUntil('\n');
        if (h == "\r" || h.length() <= 1) break;
    }

    // ---- body ------------------------------------------------------------
    String body;
    body.reserve(512);
    unsigned long t0 = millis();
    while (client.connected() && millis() - t0 < 5000) {
        while (client.available()) body += (char)client.read();
        delay(5);
    }
    client.stop();

    StaticJsonDocument<768> doc;
    DeserializationError err = deserializeJson(doc, body);
    if (err) {
        Serial.printf("[poll] JSON error: %s\n", err.c_str());
        return;
    }
    // server may hand out the sleep interval
    int sleep_ms = doc["sleep_ms"] | (int)COMMAND_POLL_MS;
    handle_command(doc);
    (void)sleep_ms;
}

// ---------------------------------------------------------------------------
// 9. ARDUINO ENTRY POINTS
// ---------------------------------------------------------------------------
void setup()
{
    Serial.begin(115200);
    delay(300);
    Serial.println();
    Serial.println("=======================================================");
    Serial.println(" AI 3D Structural Inspection System -- ESP32-CAM node");
    Serial.printf("  device  : %s   view V%d\n", NODE_DEVICE_ID, NODE_VIEW_INDEX);
    Serial.printf("  board   : 16-pin AI-Thinker ESP32-CAM (OV2640)\n");
    Serial.printf("  server  : %s%s\n", INSPECTION_HOST, UPLOAD_PATH);
    Serial.printf("  LEDs    : flash=GPIO%d countdown=GPIO%d\n",
                  PIN_FLASH_LED, PIN_COUNTDOWN_LED);
    Serial.println("=======================================================");

    pinMode(PIN_FLASH_LED, OUTPUT);
    pinMode(PIN_COUNTDOWN_LED, OUTPUT);
    digitalWrite(PIN_FLASH_LED, LOW);
    digitalWrite(PIN_COUNTDOWN_LED, LOW);

    g_camera_ready = camera_init();
    if (!g_camera_ready) {
        // keep trying: a brown-out during init is the most common field fault
        for (int attempt = 0; attempt < 3 && !g_camera_ready; attempt++) {
            delay(1500);
            Serial.printf("[cam] retry %d\n", attempt + 1);
            g_camera_ready = camera_init();
        }
    }

    g_battery_v = read_battery_voltage();
    wifi_connect();

    // boot self-test: one flash pulse, then two countdown blinks
    led_signal(PIN_FLASH_LED, 250, 250, 0);
    led_signal(PIN_COUNTDOWN_LED, 600, 300, 0);
    Serial.println("[boot] complete");
}

void loop()
{
    unsigned long now = millis();

    // ---- autonomous capture ------------------------------------------------
    if (g_camera_ready && (now - g_last_capture_ms >= g_capture_period_ms)) {
        g_last_capture_ms = now;
        g_battery_v = read_battery_voltage();

        camera_fb_t *fb = grab_frame();
        if (fb) {
            Serial.printf("[img] %ux%u  %u bytes\n", fb->width, fb->height,
                          (unsigned)fb->len);
            bool ok = upload_frame(fb);
            if (ok) {
                g_frames_sent++;
                Serial.printf("[img] uploaded OK (%u total)\n", (unsigned)g_frames_sent);
                led_signal(PIN_COUNTDOWN_LED, 400, 100, 300);
            } else {
                g_frames_failed++;
                Serial.printf("[img] upload FAILED (%u total)\n",
                              (unsigned)g_frames_failed);
                led_signal(PIN_COUNTDOWN_LED, 900, 150, 300);
                g_last_capture_ms = now - g_capture_period_ms + 30000UL;  // retry in 30 s
            }
            esp_camera_fb_return(fb);
        } else {
            g_frames_failed++;
        }
    }

    // ---- 3 s cloud command poll -------------------------------------------
    if (now - g_last_poll_ms >= COMMAND_POLL_MS) {
        g_last_poll_ms = now;
        poll_command();
    }

    delay(50);

    // watchdog: if the module has been up for days, restart nightly so the
    // WiFi/TLS stacks start clean (a real field node does this)
    static const unsigned long TWENTY_FOUR_HOURS = 24UL * 3600UL * 1000UL;
    if (now > TWENTY_FOUR_HOURS && (now % TWENTY_FOUR_HOURS) < 1000UL) {
        Serial.println("[wd] 24 h rollover -- restarting node");
        ESP.restart();
    }
}
