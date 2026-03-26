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

/**
 * Foreground service with two phases:
 *   Phase 1 (SHOW_OVERLAY): Shows the floating overlay, waits for user to tap Start
 *   Phase 2 (START_CAPTURE): Begins screen+audio capture and API loop
 *   STOP_CAPTURE: Stops capture but keeps overlay visible
 *
 * Now integrates with the real FastAPI backend, passing full PredictionResult
 * to the overlay for rich display.
 */
class CaptureService : Service() {

    companion object {
        private const val TAG = "CaptureService"
        private const val CHANNEL_ID = "deepfake_capture_channel"
        private const val NOTIFICATION_ID = 1001
        private const val FRAME_INTERVAL_MS = 1000L         // Send every 1s
        private const val HEALTH_CHECK_TIMEOUT_MS = 5000L    // Health check timeout
        private const val MAX_CONSECUTIVE_ERRORS = 5         // Auto-pause after this many errors

        const val ACTION_SHOW_OVERLAY = "com.deepfake.capture.SHOW_OVERLAY"
        const val ACTION_START_CAPTURE = "com.deepfake.capture.START_CAPTURE"
        const val ACTION_STOP_CAPTURE = "com.deepfake.capture.STOP_CAPTURE"

        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val EXTRA_SERVER_URL = "server_url"
    }

    private var mediaProjectionHelper: MediaProjectionHelper? = null
    private var videoFrameCapturer: VideoFrameCapturer? = null
    private var audioCapturer: AudioCapturer? = null
    private var apiClient: ApiClient? = null
    private var overlayManager: OverlayManager? = null
    private var captureJob: Job? = null
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    // Store MediaProjection token for later use
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
            ACTION_START_CAPTURE -> handleStartCapture(intent)
            ACTION_STOP_CAPTURE -> handleStopCapture()
        }

        return START_STICKY
    }

    // ─── Phase 1: Show overlay, store projection token ─────────

    private fun handleShowOverlay(intent: Intent?) {
        startForeground(NOTIFICATION_ID, createNotification())

        // Store the MediaProjection token
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
                Log.i(TAG, "MediaProjection token stored (code=$storedResultCode)")
            }
        }

        // Show overlay if not already visible
        if (overlayManager == null && Settings.canDrawOverlays(this)) {
            overlayManager = OverlayManager(this)

            overlayManager?.onStartCapture = { serverUrl ->
                Log.i(TAG, "Overlay: Start requested, url=$serverUrl")
                val startIntent = Intent(this, CaptureService::class.java).apply {
                    action = ACTION_START_CAPTURE
                    putExtra(EXTRA_SERVER_URL, serverUrl)
                }
                startService(startIntent)
            }

            overlayManager?.onStopCapture = {
                Log.i(TAG, "Overlay: Stop requested")
                val stopIntent = Intent(this, CaptureService::class.java).apply {
                    action = ACTION_STOP_CAPTURE
                }
                startService(stopIntent)
            }

            overlayManager?.onCloseOverlay = {
                Log.i(TAG, "Overlay: Close requested, completely stopping service")
                stopSelf() // Stop the foreground service completely
            }

            overlayManager?.show()
            Log.i(TAG, "Overlay shown, waiting for user action")
        }
    }

    // ─── Phase 2: Start capturing ──────────────────────────────

    private fun handleStartCapture(intent: Intent?) {
        val serverUrl = intent?.getStringExtra(EXTRA_SERVER_URL) ?: ""

        if (storedResultCode == Int.MIN_VALUE || storedResultData == null) {
            Log.e(TAG, "No MediaProjection token available!")
            overlayManager?.updateStatus("⚠️ No screen permission")
            return
        }

        if (serverUrl.isBlank()) {
            overlayManager?.updateStatus("⚠️ Enter server URL")
            return
        }

        // Run health check first
        overlayManager?.updateStatus("🔌 Connecting…")
        serviceScope.launch {
            val client = ApiClient(serverUrl)
            val healthy = try {
                withTimeout(HEALTH_CHECK_TIMEOUT_MS) {
                    client.checkHealth()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Health check timeout", e)
                false
            }

            if (!healthy) {
                overlayManager?.updateStatus("❌ Server unreachable")
                overlayManager?.setCapturingState(false)
                return@launch
            }

            try {
                initializeCapture(serverUrl)
                overlayManager?.setCapturingState(true)
                overlayManager?.updateStatus("📡 Capturing…")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start capture", e)
                overlayManager?.updateStatus("❌ ${e.message}")
                overlayManager?.setCapturingState(false)
            }
        }
    }

    private fun initializeCapture(serverUrl: String) {
        // Clean up any previous capture
        stopCapture()

        // Create MediaProjection only if it doesn't already exist
        if (mediaProjectionHelper == null) {
            Log.i(TAG, "Creating MediaProjection...")
            mediaProjectionHelper = MediaProjectionHelper(this)
            mediaProjectionHelper!!.createProjection(storedResultCode, storedResultData!!)
                ?: throw RuntimeException("MediaProjection returned null")
        }
        val projection = mediaProjectionHelper!!.getProjection()!!

        // Screen metrics
        val wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        val captureWidth = metrics.widthPixels
        val captureHeight = metrics.heightPixels
        val density = metrics.densityDpi

        // Initialize capturers
        videoFrameCapturer = VideoFrameCapturer(projection, captureWidth, captureHeight, density)
        audioCapturer = AudioCapturer(projection)
        apiClient = ApiClient(serverUrl)

        try { videoFrameCapturer?.start() } catch (e: Exception) {
            Log.e(TAG, "Video init failed", e); videoFrameCapturer = null
        }
        try { audioCapturer?.start(serviceScope) } catch (e: Exception) {
            Log.e(TAG, "Audio init failed", e); audioCapturer = null
        }

        startCaptureLoop()
        Log.i(TAG, "Capture started: ${captureWidth}x${captureHeight} @ $serverUrl")
    }

    // ─── Stop capture (keep overlay) ───────────────────────────

    private fun handleStopCapture() {
        stopCapture()
        overlayManager?.setCapturingState(false)
        overlayManager?.updateStatus("Idle")
        // Keep stored token and MediaProjection active so we can start again
        Log.i(TAG, "Capture stopped, overlay remains, projection kept alive")
    }

    private fun stopCapture() {
        captureJob?.cancel()
        captureJob = null
        try { videoFrameCapturer?.stop() } catch (e: Exception) { Log.w(TAG, "Error stopping video", e) }
        try { audioCapturer?.stop() } catch (e: Exception) { Log.w(TAG, "Error stopping audio", e) }
        // Do not stop mediaProjectionHelper here so we can reuse it
        videoFrameCapturer = null
        audioCapturer = null
    }

    // ─── Capture Loop ──────────────────────────────────────────

    private fun startCaptureLoop() {
        captureJob = serviceScope.launch {
            delay(500) // Stabilize

            var consecutiveErrors = 0

            while (isActive) {
                delay(FRAME_INTERVAL_MS)

                try {
                    val frame = videoFrameCapturer?.consumeLatestFrame()
                    val audio = audioCapturer?.consumeLatestSegment()

                    if (frame != null || audio != null) {
                        overlayManager?.updateStatus("🔍 Analyzing…")

                        val result = apiClient?.sendForPrediction(frame, audio)

                        if (result != null) {
                            consecutiveErrors = 0  // Reset error counter
                            overlayManager?.accumulateResult(result)
                        } else {
                            consecutiveErrors++
                            if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                                overlayManager?.updateStatus("❌ Server not responding")
                                Log.e(TAG, "Too many errors, pausing capture loop")
                                delay(5000) // Back off
                                consecutiveErrors = 0
                            } else {
                                overlayManager?.updateStatus("📡 Retrying…")
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Capture loop error", e)
                    consecutiveErrors++
                    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                        overlayManager?.updateStatus("❌ Connection lost")
                        delay(5000)
                        consecutiveErrors = 0
                    }
                }
            }
        }
    }

    // ─── Service Lifecycle ─────────────────────────────────────

    override fun onDestroy() {
        Log.i(TAG, "=== CaptureService onDestroy ===")
        stopCapture()
        try { mediaProjectionHelper?.stop() } catch (e: Exception) { Log.w(TAG, "Error stopping projection", e) }
        mediaProjectionHelper = null
        try { overlayManager?.hide() } catch (e: Exception) { Log.w(TAG, "Error hiding overlay", e) }
        overlayManager = null
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.notification_channel_desc)
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun createNotification(): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
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
