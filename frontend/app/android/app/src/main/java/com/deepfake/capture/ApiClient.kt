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
 * Primary endpoint: POST /predict_batch (sends accumulated frames as batch)
 * Legacy endpoint:  POST /predict (single frame, kept for backward compat)
 */
class ApiClient(private val serverUrl: String) {

    companion object {
        private const val TAG = "ApiClient"
    }

    private val tlsSpec = ConnectionSpec.Builder(ConnectionSpec.MODERN_TLS)
        .tlsVersions(TlsVersion.TLS_1_2, TlsVersion.TLS_1_3)
        .build()

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)   // batch may take longer
        .writeTimeout(60, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .connectionSpecs(listOf(tlsSpec, ConnectionSpec.CLEARTEXT))
        .addInterceptor { chain ->
            val request = chain.request().newBuilder()
                .addHeader("ngrok-skip-browser-warning", "true")
                .addHeader("User-Agent", "DeepFakeDetector/2.0")
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
        val personDetected: Boolean,    // was a person found in the frames?
        val securityAlert: String?,     // null, "MISMATCH_DETECTED", "LOW_CONFIDENCE"
        val threatVector: String?,      // null, "FACE_SWAP_ATTACK", "VOICE_CLONE_ATTACK"
        val analysisSource: String,     // "local" or "gemini"
        val fusionMode: String?,        // "rl_adaptive", etc.
        val videoWeight: Float?,        // RL video weight
        val audioWeight: Float?,        // RL audio weight
        val inferenceTimeMs: Float,     // server-side inference time
        val framesAnalyzed: Int = 0     // number of frames in batch
    )

    /**
     * Send a BATCH of accumulated frames + optional audio to /predict_batch.
     * This is the primary prediction method — sends all frames at once for
     * coherent multi-frame analysis by GenConViT.
     *
     * @param frames List of JPEG byte arrays (accumulated during the cycle)
     * @param audioSegment Optional audio WAV bytes
     * @return PredictionResult or null on failure
     */
    suspend fun sendBatchPrediction(
        frames: List<ByteArray>,
        audioSegment: ByteArray?
    ): PredictionResult? = withContext(Dispatchers.IO) {
        try {
            val builder = MultipartBody.Builder()
                .setType(MultipartBody.FORM)

            // Add all frames as "video_frames" parts
            for ((i, frame) in frames.withIndex()) {
                builder.addFormDataPart(
                    "video_frames",
                    "frame_$i.jpg",
                    frame.toRequestBody("image/jpeg".toMediaType())
                )
            }

            // Add audio if present
            if (audioSegment != null && audioSegment.size > 100) {
                builder.addFormDataPart(
                    "audio_segment",
                    "audio.wav",
                    audioSegment.toRequestBody("audio/wav".toMediaType())
                )
            }

            val url = serverUrl.trimEnd('/') + "/predict_batch"
            val request = Request.Builder()
                .url(url)
                .post(builder.build())
                .build()

            Log.i(TAG, "Sending batch: ${frames.size} frames + " +
                    "${if (audioSegment != null) "audio" else "no audio"} to $url")

            val response = client.newCall(request).execute()
            val body = response.body?.string()

            if (!response.isSuccessful || body == null) {
                Log.e(TAG, "Server error: ${response.code} - $body")
                return@withContext null
            }

            parsePredictionResponse(body)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send batch prediction", e)
            null
        }
    }

    /**
     * Legacy single-frame prediction (kept for backward compat).
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

            val response = client.newCall(request).execute()
            val body = response.body?.string()

            if (!response.isSuccessful || body == null) {
                Log.e(TAG, "Server error: ${response.code} - $body")
                return@withContext null
            }

            parsePredictionResponse(body)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send prediction request", e)
            null
        }
    }

    /**
     * Parse a JSON response into PredictionResult.
     */
    private fun parsePredictionResponse(body: String): PredictionResult {
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

        Log.i(TAG, "Result: ${result.prediction} " +
                "(conf=${result.confidence}, " +
                "vid=${result.videoFakeScore}, " +
                "aud=${result.audioFakeScore}, " +
                "frames=${result.framesAnalyzed}, " +
                "time=${result.inferenceTimeMs}ms)")

        return result
    }

    /**
     * Check if the server is reachable.
     */
    suspend fun checkHealth(): Boolean = withContext(Dispatchers.IO) {
        try {
            val url = serverUrl.trimEnd('/') + "/health"
            val request = Request.Builder().url(url).get().build()
            val response = client.newCall(request).execute()
            val ok = response.isSuccessful
            Log.i(TAG, "Health check: ${if (ok) "OK" else "FAIL (${response.code})"}")
            ok
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Health check failed", e)
            false
        }
    }

    /**
     * Lightweight person presence check — sends a single frame to /check_person.
     * Used for auto-start scanning (no model inference on server).
     */
    suspend fun checkPerson(frameJpeg: ByteArray): Boolean = withContext(Dispatchers.IO) {
        try {
            val body = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "frame", "scan.jpg",
                    frameJpeg.toRequestBody("image/jpeg".toMediaType())
                )
                .build()

            val url = serverUrl.trimEnd('/') + "/check_person"
            val request = Request.Builder().url(url).post(body).build()
            val response = client.newCall(request).execute()
            val respBody = response.body?.string()

            if (!response.isSuccessful || respBody == null) {
                Log.w(TAG, "check_person failed: ${response.code}")
                return@withContext false
            }

            val json = org.json.JSONObject(respBody)
            val detected = json.optBoolean("person_detected", false)
            val type = json.optString("type", "none")
            Log.i(TAG, "Person check: detected=$detected type=$type")
            detected
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "checkPerson failed", e)
            false
        }
    }
}
