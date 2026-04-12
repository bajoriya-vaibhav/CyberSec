package com.deepfake.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import kotlinx.coroutines.*
import java.util.concurrent.CancellationException

/**
 * Foreground service for continuous deepfake streaming analysis.
 *
 * Three modes:
 *   IDLE     → Overlay shown, no capture running
 *   SCANNING → Auto-mode: captures a frame every 2s, POSTs to /check_person
 *              When person detected → auto-transitions to STREAMING
 *   STREAMING → Full WebSocket streaming to /ws/analyze
 *               If server reports no person for 3 consecutive results → auto-pause → SCANNING
 *
 * Manual start/stop always available regardless of auto mode.
 */
class CaptureService : Service() {

    companion object {
        private const val TAG = "CaptureService"
        private const val CHANNEL_ID = "deepfake_capture_channel"
        private const val NOTIFICATION_ID = 1001
        private const val FRAME_INTERVAL_MS = 3000L       // Streaming: send frame every 3s
        private const val SCAN_INTERVAL_MS = 2500L        // Scanning: check every 2.5s
        private const val NO_PERSON_THRESHOLD = 3          // Stop after N consecutive no-person results

        const val ACTION_SHOW_OVERLAY = "com.deepfake.capture.SHOW_OVERLAY"
        const val ACTION_START_CAPTURE = "com.deepfake.capture.START_CAPTURE"
        const val ACTION_STOP_CAPTURE = "com.deepfake.capture.STOP_CAPTURE"

        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val EXTRA_SERVER_URL = "server_url"

        private const val PREFS_NAME = "deepfake_prefs"
        private const val KEY_SERVER_URL = "server_url"
        private const val KEY_ANALYSIS_DURATION = "analysis_duration"
        private const val KEY_AUTO_MODE = "auto_mode"
    }

    private enum class CaptureState { IDLE, SCANNING, STREAMING }

    private var state = CaptureState.IDLE
    private var mediaProjectionHelper: MediaProjectionHelper? = null
    private var videoFrameCapturer: VideoFrameCapturer? = null
    private var audioCapturer: AudioCapturer? = null
    private var streamingClient: StreamingClient? = null
    private var overlayManager: OverlayManager? = null
    private var captureJob: Job? = null
    private var scanJob: Job? = null
    private var consecutiveNoPersonResults = 0
    private var manualMode = false  // true when user manually started (bypasses auto-stop)

    private val exceptionHandler = CoroutineExceptionHandler { _, throwable ->
        Log.e(TAG, "Uncaught coroutine exception", throwable)
        overlayManager?.updateStatus("❌ Error: ${throwable.message?.take(40)}")
    }
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main + exceptionHandler)

    private var storedResultCode: Int = Int.MIN_VALUE
    private var storedResultData: Intent? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        Log.i(TAG, "=== CaptureService created ===")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: ACTION_SHOW_OVERLAY
        Log.i(TAG, "onStartCommand action=$action")

        when (action) {
            ACTION_SHOW_OVERLAY -> handleShowOverlay(intent)
            ACTION_START_CAPTURE -> handleStartCapture(isManual = true)
            ACTION_STOP_CAPTURE -> handleStopCapture(isManual = true)
        }

        return START_STICKY
    }

    // ─── Phase 1: Show overlay, optionally start scanning ──────

    private fun handleShowOverlay(intent: Intent?) {
        startForeground(NOTIFICATION_ID, createNotification())

        if (intent != null) {
            val code = intent.getIntExtra(EXTRA_RESULT_CODE, Int.MIN_VALUE)
            if (code != Int.MIN_VALUE) {
                storedResultCode = code
                storedResultData = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_RESULT_DATA)
                }
                Log.i(TAG, "MediaProjection token stored")
            }
        }

        if (overlayManager == null && Settings.canDrawOverlays(this)) {
            overlayManager = OverlayManager(this)

            overlayManager?.onStartCapture = { _ ->
                Log.i(TAG, "Manual start requested")
                handleStartCapture(isManual = true)
            }

            overlayManager?.onStopCapture = {
                Log.i(TAG, "Manual stop requested")
                handleStopCapture(isManual = true)
            }

            overlayManager?.onCloseOverlay = {
                Log.i(TAG, "Close overlay → stop service")
                stopSelf()
            }

            overlayManager?.onAudioToggleChanged = { audioEnabled ->
                val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                val duration = prefs.getInt(KEY_ANALYSIS_DURATION, 15)
                streamingClient?.sendConfig(duration, audioEnabled)
            }

            overlayManager?.onAutoToggleChanged = { autoEnabled ->
                Log.i(TAG, "Auto mode: $autoEnabled")
                if (autoEnabled && state == CaptureState.IDLE) {
                    startScanning()
                } else if (!autoEnabled && state == CaptureState.SCANNING) {
                    stopScanning()
                    overlayManager?.updateStatus("Ready")
                    state = CaptureState.IDLE
                }
            }

            overlayManager?.show()
            Log.i(TAG, "Overlay shown")

            // Auto-start scanning if auto mode is enabled
            val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val autoMode = prefs.getBoolean(KEY_AUTO_MODE, true)
            if (autoMode) {
                serviceScope.launch {
                    delay(1000) // Let overlay settle
                    if (state == CaptureState.IDLE && overlayManager?.isAutoEnabled == true) {
                        startScanning()
                    }
                }
            }
        }
    }

    // ─── Scanning Mode (lightweight person detection) ──────────

    private fun startScanning() {
        if (state == CaptureState.STREAMING) return // Don't scan while streaming
        if (scanJob?.isActive == true) return

        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val serverUrl = prefs.getString(KEY_SERVER_URL, "") ?: ""

        if (storedResultCode == Int.MIN_VALUE || storedResultData == null) {
            overlayManager?.updateStatus("⚠️ No screen permission")
            return
        }
        if (serverUrl.isBlank()) {
            overlayManager?.updateStatus("⚠️ Set server URL in app")
            return
        }

        state = CaptureState.SCANNING

        // Ensure video capture is initialized for scanning
        if (videoFrameCapturer == null) {
            try {
                initializeVideoCapture()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to init video for scanning", e)
                overlayManager?.updateStatus("❌ Screen capture error")
                state = CaptureState.IDLE
                return
            }
        }

        overlayManager?.updateStatus("👁 Scanning for person…")
        overlayManager?.setScanningState(true)
        Log.i(TAG, "Scanning started")

        scanJob = serviceScope.launch {
            val apiClient = ApiClient(serverUrl)

            while (isActive && state == CaptureState.SCANNING) {
                delay(SCAN_INTERVAL_MS)

                val frame = videoFrameCapturer?.consumeLatestFrame()
                if (frame == null) {
                    continue
                }

                try {
                    val personDetected = apiClient.checkPerson(frame)

                    if (personDetected) {
                        Log.i(TAG, "Person detected! Auto-starting streaming")
                        overlayManager?.updateStatus("👤 Person detected!")
                        delay(300) // Brief notification
                        handleStartCapture(isManual = false)
                        return@launch // Exit scan loop
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.w(TAG, "Scan check failed: ${e.message}")
                }
            }
        }
    }

    private fun stopScanning() {
        scanJob?.cancel()
        scanJob = null
        overlayManager?.setScanningState(false)
    }

    // ─── Streaming Mode ────────────────────────────────────────

    private fun handleStartCapture(isManual: Boolean) {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val serverUrl = prefs.getString(KEY_SERVER_URL, "") ?: ""
        val duration = prefs.getInt(KEY_ANALYSIS_DURATION, 15)
        val audioEnabled = overlayManager?.isAudioEnabled ?: false

        if (storedResultCode == Int.MIN_VALUE || storedResultData == null) {
            overlayManager?.updateStatus("⚠️ No screen permission")
            return
        }
        if (serverUrl.isBlank()) {
            overlayManager?.updateStatus("⚠️ Set server URL in app")
            return
        }

        manualMode = isManual
        consecutiveNoPersonResults = 0
        stopScanning()

        overlayManager?.updateStatus("🔌 Connecting…")
        serviceScope.launch {
            // Health check
            val healthy = try {
                withTimeout(5000) { ApiClient(serverUrl).checkHealth() }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { false }

            if (!healthy) {
                overlayManager?.updateStatus("❌ Server unreachable")
                overlayManager?.setCapturingState(false)
                if (overlayManager?.isAutoEnabled == true) startScanning()
                return@launch
            }

            try {
                // Ensure video + audio capture running
                if (videoFrameCapturer == null) initializeVideoCapture()
                if (audioCapturer == null) initializeAudioCapture()

                state = CaptureState.STREAMING

                streamingClient = StreamingClient(serverUrl).apply {
                    onConnected = {
                        serviceScope.launch(Dispatchers.Main) {
                            overlayManager?.setCapturingState(true)
                            overlayManager?.updateStatus("📡 Streaming…")
                            sendConfig(duration, audioEnabled)
                        }
                    }
                    onResult = { result ->
                        serviceScope.launch(Dispatchers.Main) {
                            handleServerResult(result)
                        }
                    }
                    onStatus = { buffered, secLeft ->
                        serviceScope.launch(Dispatchers.Main) {
                            overlayManager?.updateStreamingProgress(
                                buffered, secLeft, this@apply.framesSent
                            )
                        }
                    }
                    onAnalyzing = { _ -> /* silent */ }
                    onDisconnected = { reason ->
                        serviceScope.launch(Dispatchers.Main) {
                            overlayManager?.updateStatus("❌ Disconnected")
                            overlayManager?.setCapturingState(false)
                            stopStreaming()
                            if (overlayManager?.isAutoEnabled == true) startScanning()
                        }
                    }
                    onError = { msg ->
                        serviceScope.launch(Dispatchers.Main) {
                            overlayManager?.updateStatus("❌ $msg")
                        }
                    }
                }
                streamingClient?.connect()
                startStreamingLoop()

            } catch (e: Exception) {
                Log.e(TAG, "Failed to start streaming", e)
                overlayManager?.updateStatus("❌ ${e.message?.take(40)}")
                overlayManager?.setCapturingState(false)
                state = CaptureState.IDLE
            }
        }
    }

    /**
     * Handle a result from the server.
     * If person_detected is false for N consecutive results → auto-pause.
     */
    private fun handleServerResult(result: ApiClient.PredictionResult) {
        if (!result.personDetected) {
            consecutiveNoPersonResults++
            Log.i(TAG, "No person detected ($consecutiveNoPersonResults/$NO_PERSON_THRESHOLD)")

            if (!manualMode && consecutiveNoPersonResults >= NO_PERSON_THRESHOLD) {
                Log.i(TAG, "Auto-pausing: no person for $NO_PERSON_THRESHOLD cycles")
                overlayManager?.showNoPersonNotification()
                handleStopCapture(isManual = false)
                return
            }
        } else {
            consecutiveNoPersonResults = 0
        }

        overlayManager?.showBatchResult(result)
    }

    private fun startStreamingLoop() {
        captureJob = serviceScope.launch {
            delay(500)
            while (isActive && state == CaptureState.STREAMING) {
                delay(FRAME_INTERVAL_MS)
                val frame = videoFrameCapturer?.consumeLatestFrame()
                if (frame != null) {
                    streamingClient?.sendFrame(frame)
                }
                if (overlayManager?.isAudioEnabled == true) {
                    val audio = audioCapturer?.consumeLatestSegment()
                    if (audio != null) {
                        streamingClient?.sendAudio(audio)
                    }
                }
            }
        }
    }

    // ─── Stop ──────────────────────────────────────────────────

    private fun handleStopCapture(isManual: Boolean) {
        stopStreaming()
        overlayManager?.setCapturingState(false)

        if (isManual) {
            overlayManager?.updateStatus("Idle")
            manualMode = false
            state = CaptureState.IDLE
            // If auto mode, resume scanning after manual stop
            if (overlayManager?.isAutoEnabled == true) {
                serviceScope.launch {
                    delay(2000)
                    if (state == CaptureState.IDLE) startScanning()
                }
            }
        } else {
            // Auto-stop: resume scanning
            state = CaptureState.IDLE
            if (overlayManager?.isAutoEnabled == true) {
                serviceScope.launch {
                    delay(3000) // Wait a bit before rescanning
                    if (state == CaptureState.IDLE) startScanning()
                }
            }
        }
    }

    private fun stopStreaming() {
        captureJob?.cancel()
        captureJob = null
        streamingClient?.disconnect()
        streamingClient = null
        state = CaptureState.IDLE
    }

    // ─── Initialization Helpers ────────────────────────────────

    private fun initializeVideoCapture() {
        if (mediaProjectionHelper == null) {
            mediaProjectionHelper = MediaProjectionHelper(this)
            mediaProjectionHelper!!.createProjection(storedResultCode, storedResultData!!)
                ?: throw RuntimeException("MediaProjection returned null")
        }
        val projection = mediaProjectionHelper!!.getProjection()!!

        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)

        videoFrameCapturer = VideoFrameCapturer(
            projection, metrics.widthPixels, metrics.heightPixels, metrics.densityDpi
        )
        try { videoFrameCapturer?.start() } catch (e: Exception) {
            Log.e(TAG, "Video init failed", e)
            videoFrameCapturer = null
            throw e
        }
    }

    private fun initializeAudioCapture() {
        val projection = mediaProjectionHelper?.getProjection() ?: return
        audioCapturer = AudioCapturer(projection)
        try { audioCapturer?.start(serviceScope) } catch (e: Exception) {
            Log.e(TAG, "Audio init failed", e)
            audioCapturer = null
        }
    }

    // ─── Service Lifecycle ─────────────────────────────────────

    override fun onDestroy() {
        Log.i(TAG, "=== CaptureService onDestroy ===")
        stopScanning()
        stopStreaming()
        try { videoFrameCapturer?.stop() } catch (e: Exception) {}
        try { audioCapturer?.stop() } catch (e: Exception) {}
        videoFrameCapturer = null
        audioCapturer = null
        try { mediaProjectionHelper?.stop() } catch (e: Exception) {}
        mediaProjectionHelper = null
        try { overlayManager?.hide() } catch (e: Exception) {}
        overlayManager = null
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply { description = getString(R.string.notification_channel_desc) }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun createNotification(): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
