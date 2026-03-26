package com.deepfake.capture

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.ConnectionSpec
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.TlsVersion
import org.json.JSONObject
import java.util.concurrent.CancellationException
import java.util.concurrent.TimeUnit

/**
 * REST API client for the DeepFake Detection backend.
 *
 * Sends captured frames (JPEG) and audio segments (WAV) to:
 *   POST <serverUrl>/predict  (multipart/form-data)
 *
 * Expected server response:
 * {
 *   "prediction": "Real" | "Fake" | "Suspicious",
 *   "confidence": 0.95,
 *   "fake_probability": 0.87,
 *   "video_fake_score": 0.82,
 *   "audio_fake_score": 0.93,
 *   "security_alert": null | "MISMATCH_DETECTED" | "LOW_CONFIDENCE",
 *   "threat_vector": null | "FACE_SWAP_ATTACK" | "VOICE_CLONE_ATTACK",
 *   "analysis_source": "local" | "gemini",
 *   "fusion_mode": "rl_adaptive",
 *   "rl_weights": { "video_weight": 0.9, "audio_weight": 0.1 },
 *   "inference_time_ms": 245.3
 * }
 */
class ApiClient(private val serverUrl: String) {

    companion object {
        private const val TAG = "ApiClient"
    }

    // Support both TLS 1.2 (Android 8.x) and TLS 1.3 (Android 10+)
    private val tlsSpec = ConnectionSpec.Builder(ConnectionSpec.MODERN_TLS)
        .tlsVersions(TlsVersion.TLS_1_2, TlsVersion.TLS_1_3)
        .build()

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .connectionSpecs(listOf(tlsSpec, ConnectionSpec.CLEARTEXT))
        .addInterceptor { chain ->
            val request = chain.request().newBuilder()
                .addHeader("ngrok-skip-browser-warning", "true")
                .addHeader("User-Agent", "DeepFakeDetector/1.0")
                .build()
            chain.proceed(request)
        }
        .build()

    /**
     * Full prediction result from the server.
     */
    data class PredictionResult(
        val prediction: String,         // "Real", "Fake", "Suspicious"
        val confidence: Float,          // 0.0 – 1.0
        val fakeProbability: Float,     // combined fake score
        val videoFakeScore: Float?,     // individual video score
        val audioFakeScore: Float?,     // individual audio score
        val securityAlert: String?,     // null, "MISMATCH_DETECTED", "LOW_CONFIDENCE"
        val threatVector: String?,      // null, "FACE_SWAP_ATTACK", "VOICE_CLONE_ATTACK"
        val analysisSource: String,     // "local" or "gemini"
        val fusionMode: String?,        // "rl_adaptive", "security_first", etc.
        val videoWeight: Float?,        // RL video weight
        val audioWeight: Float?,        // RL audio weight
        val inferenceTimeMs: Float      // server-side inference time
    )

    /**
     * Sends a video frame and/or audio segment to the backend for analysis.
     * Either parameter can be null if that modality is unavailable.
     * Returns PredictionResult or null on failure.
     */
    suspend fun sendForPrediction(
        videoFrame: ByteArray?,
        audioSegment: ByteArray?
    ): PredictionResult? = withContext(Dispatchers.IO) {
        try {
            val builder = MultipartBody.Builder()
                .setType(MultipartBody.FORM)

            if (videoFrame != null) {
                builder.addFormDataPart(
                    "video_frame",
                    "frame.jpg",
                    videoFrame.toRequestBody("image/jpeg".toMediaType())
                )
            }

            if (audioSegment != null && audioSegment.size > 100) {
                builder.addFormDataPart(
                    "audio_segment",
                    "audio.wav",
                    audioSegment.toRequestBody("audio/wav".toMediaType())
                )
            }

            val url = serverUrl.trimEnd('/') + "/predict"
            val request = Request.Builder()
                .url(url)
                .post(builder.build())
                .build()

            Log.d(TAG, "Sending prediction request to $url")

            val response = client.newCall(request).execute()
            val body = response.body?.string()

            if (!response.isSuccessful || body == null) {
                Log.e(TAG, "Server error: ${response.code} - $body")
                return@withContext null
            }

            val json = JSONObject(body)

            val rlWeights = json.optJSONObject("rl_weights")

            val result = PredictionResult(
                prediction = json.optString("prediction", "Unknown"),
                confidence = json.optDouble("confidence", 0.0).toFloat(),
                fakeProbability = json.optDouble("fake_probability", 0.5).toFloat(),
                videoFakeScore = if (json.isNull("video_fake_score")) null
                                 else json.optDouble("video_fake_score").toFloat(),
                audioFakeScore = if (json.isNull("audio_fake_score")) null
                                 else json.optDouble("audio_fake_score").toFloat(),
                securityAlert = json.optString("security_alert", "").ifBlank { null },
                threatVector = json.optString("threat_vector", "").ifBlank { null },
                analysisSource = json.optString("analysis_source", "local"),
                fusionMode = json.optString("fusion_mode", "").ifBlank { null },
                videoWeight = rlWeights?.optDouble("video_weight")?.toFloat(),
                audioWeight = rlWeights?.optDouble("audio_weight")?.toFloat(),
                inferenceTimeMs = json.optDouble("inference_time_ms", 0.0).toFloat()
            )

            Log.i(TAG, "Prediction: ${result.prediction} " +
                    "(conf=${result.confidence}, " +
                    "vid=${result.videoFakeScore}, " +
                    "aud=${result.audioFakeScore}, " +
                    "time=${result.inferenceTimeMs}ms)")

            result
        } catch (e: CancellationException) {
            throw e  // Don't swallow coroutine cancellation
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send prediction request", e)
            null
        }
    }

    /**
     * Check if the server is reachable and models are loaded.
     * Returns true if health check succeeds.
     */
    suspend fun checkHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val url = serverUrl.trimEnd('/') + "/health"
            val request = Request.Builder().url(url).get().build()
            val response = client.newCall(request).execute()
            val body = response.body?.string()
            val ok = response.isSuccessful
            Log.i(TAG, "Health check: ${if (ok) "OK" else "FAIL (${response.code})"} body=${body?.take(100)}")
            ok
        } catch (e: CancellationException) {
            throw e  // Don't swallow coroutine cancellation
        } catch (e: Exception) {
            Log.e(TAG, "Health check failed", e)
            false
        }
    }
}
