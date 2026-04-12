package com.deepfake.capture

import android.util.Log
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * WebSocket client for continuous streaming to /ws/analyze.
 *
 * Sending frames and receiving results are fully decoupled:
 *   - sendFrame() / sendAudio() fire-and-forget (never block)
 *   - Results arrive asynchronously via onResult callback
 *   - Status updates arrive via onStatus callback
 *
 * Binary protocol:
 *   0x01 + JPEG bytes = video frame
 *   0x02 + WAV bytes  = audio chunk
 *
 * Text protocol:
 *   {"type":"config", "duration":15, "audio_enabled":false}  → Server
 *   {"type":"result", "prediction":"Fake", ...}              ← Server
 *   {"type":"status", "frames_buffered":12, ...}             ← Server
 *   {"type":"analyzing", "frames":15}                        ← Server
 */
class StreamingClient(private val serverUrl: String) {

    companion object {
        private const val TAG = "StreamingClient"
        private const val VIDEO_MARKER: Byte = 0x01
        private const val AUDIO_MARKER: Byte = 0x02
    }

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)  // No timeout for WebSocket
        .connectTimeout(10, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)     // Keep-alive
        .addInterceptor { chain ->
            val request = chain.request().newBuilder()
                .addHeader("ngrok-skip-browser-warning", "true")
                .addHeader("User-Agent", "DeepGuard-Stream/2.0")
                .build()
            chain.proceed(request)
        }
        .build()

    private var webSocket: WebSocket? = null
    private var isConnected = false
    var framesSent = 0
        private set

    // ─── Callbacks (set by CaptureService) ──────────────────────

    var onResult: ((ApiClient.PredictionResult) -> Unit)? = null
    var onStatus: ((framesBuffered: Int, secondsLeft: Int) -> Unit)? = null
    var onAnalyzing: ((frameCount: Int) -> Unit)? = null
    var onConnected: (() -> Unit)? = null
    var onDisconnected: ((reason: String) -> Unit)? = null
    var onError: ((message: String) -> Unit)? = null

    // ─── Connect ────────────────────────────────────────────────

    fun connect() {
        val wsUrl = serverUrl
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            .trimEnd('/') + "/ws/analyze"

        Log.i(TAG, "Connecting to $wsUrl")

        val request = Request.Builder().url(wsUrl).build()
        webSocket = client.newWebSocket(request, WsListener())
    }

    // ─── Send (fire-and-forget, never blocks) ───────────────────

    fun sendFrame(jpeg: ByteArray): Boolean {
        if (!isConnected || webSocket == null) return false
        val buf = ByteArray(1 + jpeg.size)
        buf[0] = VIDEO_MARKER
        System.arraycopy(jpeg, 0, buf, 1, jpeg.size)
        val ok = webSocket?.send(buf.toByteString()) ?: false
        if (ok) framesSent++
        return ok
    }

    fun sendAudio(wav: ByteArray): Boolean {
        if (!isConnected || webSocket == null) return false
        val buf = ByteArray(1 + wav.size)
        buf[0] = AUDIO_MARKER
        System.arraycopy(wav, 0, buf, 1, wav.size)
        return webSocket?.send(buf.toByteString()) ?: false
    }

    fun sendConfig(duration: Int, audioEnabled: Boolean) {
        val json = JSONObject().apply {
            put("type", "config")
            put("duration", duration)
            put("audio_enabled", audioEnabled)
        }
        webSocket?.send(json.toString())
        Log.i(TAG, "Config sent: duration=${duration}s, audio=$audioEnabled")
    }

    // ─── Disconnect ─────────────────────────────────────────────

    fun disconnect() {
        isConnected = false
        webSocket?.close(1000, "User stopped")
        webSocket = null
        framesSent = 0
        Log.i(TAG, "Disconnected")
    }

    fun isConnected(): Boolean = isConnected

    // ─── WebSocket Listener ─────────────────────────────────────

    private inner class WsListener : WebSocketListener() {

        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.i(TAG, "WebSocket connected")
            isConnected = true
            framesSent = 0
            onConnected?.invoke()
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            try {
                val json = JSONObject(text)
                when (json.optString("type")) {
                    "result" -> {
                        val result = parseResult(json)
                        Log.i(TAG, "Result: ${result.prediction} (${result.confidence})")
                        onResult?.invoke(result)
                    }
                    "status" -> {
                        val buffered = json.optInt("frames_buffered", 0)
                        val secLeft = json.optDouble("seconds_until_analysis", 0.0).toInt()
                        onStatus?.invoke(buffered, secLeft)
                    }
                    "analyzing" -> {
                        val frames = json.optInt("frames", 0)
                        Log.i(TAG, "Server analyzing $frames frames...")
                        onAnalyzing?.invoke(frames)
                    }
                    "error" -> {
                        val msg = json.optString("message", "Unknown error")
                        Log.e(TAG, "Server error: $msg")
                        onError?.invoke(msg)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to parse message: $text", e)
            }
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.e(TAG, "WebSocket failure: ${t.message}", t)
            isConnected = false
            onDisconnected?.invoke(t.message ?: "Connection failed")
        }

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            Log.i(TAG, "WebSocket closing: $code $reason")
            webSocket.close(code, reason)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            Log.i(TAG, "WebSocket closed: $code $reason")
            isConnected = false
            onDisconnected?.invoke(reason)
        }
    }

    // ─── Parse server result JSON ───────────────────────────────

    private fun parseResult(json: JSONObject): ApiClient.PredictionResult {
        val rlWeights = json.optJSONObject("rl_weights")
        return ApiClient.PredictionResult(
            prediction = json.optString("prediction", "Unknown"),
            confidence = json.optDouble("confidence", 0.0).toFloat(),
            fakeProbability = json.optDouble("fake_probability", 0.5).toFloat(),
            videoFakeScore = if (json.isNull("video_fake_score")) null
                             else json.optDouble("video_fake_score").toFloat(),
            audioFakeScore = if (json.isNull("audio_fake_score")) null
                             else json.optDouble("audio_fake_score").toFloat(),
            personDetected = json.optBoolean("person_detected", true),
            securityAlert = json.optString("security_alert", "").ifBlank { null },
            threatVector = json.optString("threat_vector", "").ifBlank { null },
            analysisSource = json.optString("analysis_source", "local"),
            fusionMode = json.optString("fusion_mode", "").ifBlank { null },
            videoWeight = rlWeights?.optDouble("video_weight")?.toFloat(),
            audioWeight = rlWeights?.optDouble("audio_weight")?.toFloat(),
            inferenceTimeMs = json.optDouble("inference_time_ms", 0.0).toFloat(),
            framesAnalyzed = json.optInt("frames_analyzed", 0)
        )
    }
}
