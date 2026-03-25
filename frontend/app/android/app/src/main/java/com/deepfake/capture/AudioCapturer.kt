package com.deepfake.capture

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.projection.MediaProjection
import android.util.Log
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Captures internal audio using AudioPlaybackCapture (API 29+).
 * Falls back to microphone recording if playback capture is unavailable.
 * Records 3-second WAV segments.
 */
class AudioCapturer(
    private val mediaProjection: MediaProjection?
) {
    companion object {
        private const val TAG = "AudioCapturer"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null
    private var isUsingPlaybackCapture = false

    private val pcmBuffer = ByteArrayOutputStream()
    private val bufferLock = Any()

    fun start(scope: CoroutineScope) {
        val minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid buffer size: $minBufferSize")
            return
        }
        val bufferSize = maxOf(minBufferSize, SAMPLE_RATE * 2)

        // Try playback capture first, then microphone fallback
        audioRecord = try {
            if (mediaProjection != null) {
                createPlaybackCaptureRecord(bufferSize)
            } else {
                throw IllegalStateException("No MediaProjection available")
            }
        } catch (e: Exception) {
            Log.w(TAG, "AudioPlaybackCapture failed: ${e.message}, trying microphone fallback")
            try {
                createMicrophoneRecord(bufferSize)
            } catch (e2: Exception) {
                Log.e(TAG, "Microphone fallback also failed: ${e2.message}")
                null
            }
        }

        if (audioRecord == null) {
            Log.e(TAG, "Could not create any AudioRecord")
            return
        }

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord state=${audioRecord?.state}, not initialized. Releasing.")
            audioRecord?.release()
            audioRecord = null
            return
        }

        try {
            audioRecord!!.startRecording()
            Log.i(TAG, "Audio capture started (playbackCapture=$isUsingPlaybackCapture)")
        } catch (e: Exception) {
            Log.e(TAG, "startRecording() failed", e)
            audioRecord?.release()
            audioRecord = null
            return
        }

        captureJob = scope.launch(Dispatchers.IO) {
            val buffer = ShortArray(2048)

            while (isActive) {
                try {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: break

                    if (read > 0) {
                        synchronized(bufferLock) {
                            for (i in 0 until read) {
                                // Write shorts as little endian to our PCM buffer
                                pcmBuffer.write(buffer[i].toInt() and 0xFF)
                                pcmBuffer.write((buffer[i].toInt() shr 8) and 0xFF)
                            }
                        }
                    } else if (read < 0) {
                        Log.e(TAG, "AudioRecord error: $read")
                        break
                    } else {
                        // read == 0 means no audio data currently available (silence / no playback)
                        delay(10)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error recording audio segment", e)
                    delay(1000) // Wait before retrying
                }
            }
        }
    }

    private fun createPlaybackCaptureRecord(bufferSize: Int): AudioRecord {
        val config = AudioPlaybackCaptureConfiguration.Builder(mediaProjection!!)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .addMatchingUsage(AudioAttributes.USAGE_GAME)
            .addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
            .build()

        isUsingPlaybackCapture = true

        return AudioRecord.Builder()
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(CHANNEL_CONFIG)
                    .setEncoding(AUDIO_FORMAT)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .setAudioPlaybackCaptureConfig(config)
            .build()
    }

    @Suppress("MissingPermission")
    private fun createMicrophoneRecord(bufferSize: Int): AudioRecord {
        isUsingPlaybackCapture = false
        Log.i(TAG, "Creating microphone AudioRecord (fallback)")
        return AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )
    }

    fun consumeLatestSegment(): ByteArray? {
        val pcmBytes = synchronized(bufferLock) {
            if (pcmBuffer.size() == 0) return null
            val data = pcmBuffer.toByteArray()
            pcmBuffer.reset()
            data
        }
        return pcmToWav(pcmBytes)
    }

    fun stop() {
        captureJob?.cancel()
        captureJob = null
        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping AudioRecord", e)
        }
        audioRecord?.release()
        audioRecord = null
        synchronized(bufferLock) {
            pcmBuffer.reset()
        }
        Log.i(TAG, "Audio capture stopped")
    }

    private fun pcmToWav(pcmData: ByteArray): ByteArray {
        val pcmBytes = pcmData.size
        val totalSize = 44 + pcmBytes

        val output = ByteArrayOutputStream(totalSize)
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)

        header.put("RIFF".toByteArray())
        header.putInt(totalSize - 8)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)
        header.putShort(1)
        header.putInt(SAMPLE_RATE)
        header.putInt(SAMPLE_RATE * 2)
        header.putShort(2)
        header.putShort(16)
        header.put("data".toByteArray())
        header.putInt(pcmBytes)

        output.write(header.array())
        output.write(pcmData)

        return output.toByteArray()
    }
}
