package com.deepfake.capture

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Compact floating overlay for DeepGuard.
 *
 * Features:
 *   - Auto toggle: scans for person and auto-starts/stops streaming
 *   - Audio toggle: enables/disables audio analysis
 *   - Manual start/stop always available
 *   - "No person detected" notification
 *   - Async result display (never blocks streaming)
 */
class OverlayManager(private val context: Context) {

    companion object {
        private const val TAG = "OverlayManager"
        private const val PREFS_NAME = "deepfake_prefs"
        private const val KEY_AUDIO_ENABLED = "audio_enabled"
        private const val KEY_AUTO_MODE = "auto_mode"
    }

    private var windowManager: WindowManager? = null
    private var bubbleView: View? = null
    private var bubbleParams: WindowManager.LayoutParams? = null
    private var panelView: View? = null
    private var panelParams: WindowManager.LayoutParams? = null

    // UI references
    private var statusText: TextView? = null
    private var statusDot: View? = null
    private var streamInfoText: TextView? = null
    private var verdictEmoji: TextView? = null
    private var verdictLabel: TextView? = null
    private var confidenceLabel: TextView? = null
    private var videoScoreText: TextView? = null
    private var audioScoreText: TextView? = null
    private var detailsText: TextView? = null
    private var historyStrip: TextView? = null
    private var startBtn: TextView? = null
    private var stopBtn: TextView? = null
    private var audioToggleBtn: TextView? = null
    private var autoToggleBtn: TextView? = null
    private var resultSection: LinearLayout? = null
    private var streamSection: LinearLayout? = null

    private var isExpanded = false
    private var isCapturing = false
    private var isScanning = false

    var isAudioEnabled = false
        private set
    var isAutoEnabled = true
        private set

    // Callbacks
    var onStartCapture: ((serverUrl: String) -> Unit)? = null
    var onStopCapture: (() -> Unit)? = null
    var onCloseOverlay: (() -> Unit)? = null
    var onAudioToggleChanged: ((enabled: Boolean) -> Unit)? = null
    var onAutoToggleChanged: ((enabled: Boolean) -> Unit)? = null

    // History
    private data class CycleResult(val verdict: String, val confidence: Float, val num: Int)
    private var cycleNumber = 0
    private val history = mutableListOf<CycleResult>()

    private fun dp(v: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), context.resources.displayMetrics).toInt()

    // ─── Public API ────────────────────────────────────────────

    fun show() {
        if (bubbleView != null) return
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        isAudioEnabled = prefs.getBoolean(KEY_AUDIO_ENABLED, false)
        isAutoEnabled = prefs.getBoolean(KEY_AUTO_MODE, true)
        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        showBubble()
    }

    fun hide() {
        removeBubble(); removePanel(); windowManager = null
    }

    fun updateStatus(status: String) {
        panelView?.post {
            statusText?.text = status
        }
    }

    fun setScanningState(scanning: Boolean) {
        isScanning = scanning
        panelView?.post {
            if (scanning) {
                streamSection?.visibility = View.VISIBLE
                streamInfoText?.text = "👁 Scanning for person…"
                statusDot?.background = makeDot(Color.parseColor("#FFA726"))
                // Show pulsing dot effect
                statusDot?.animate()?.alpha(0.3f)?.setDuration(600)?.withEndAction {
                    statusDot?.animate()?.alpha(1f)?.setDuration(600)?.start()
                }?.start()
            } else {
                statusDot?.animate()?.cancel()
                statusDot?.alpha = 1f
            }
            // Allow manual start during scanning
            startBtn?.apply { isEnabled = true; alpha = 1f }
            stopBtn?.apply { isEnabled = false; alpha = 0.4f }
        }
    }

    fun updateStreamingProgress(framesBuffered: Int, secondsUntilAnalysis: Int, totalFramesSent: Int) {
        panelView?.post {
            streamSection?.visibility = View.VISIBLE
            val secText = if (secondsUntilAnalysis > 0) "analyzing in ${secondsUntilAnalysis}s" else "analyzing…"
            streamInfoText?.text = "$framesBuffered buffered • $totalFramesSent sent • $secText"
        }
    }

    fun showNoPersonNotification() {
        panelView?.post {
            resultSection?.visibility = View.GONE
            streamSection?.visibility = View.VISIBLE
            streamInfoText?.text = "⚠ No person detected — pausing"
            statusText?.text = "⚠ No person"
            statusText?.setTextColor(Color.parseColor("#FFA726"))
            statusDot?.background = makeDot(Color.parseColor("#FFA726"))
        }
        bubbleView?.post {
            (bubbleView as? TextView)?.apply {
                text = "⚠️"
                setTextColor(Color.parseColor("#FFA726"))
            }
        }
    }

    fun showBatchResult(result: ApiClient.PredictionResult) {
        cycleNumber++
        val verdict = result.prediction
        val confidence = result.confidence
        val isReal = verdict.equals("Real", true)
        val isFake = verdict.equals("Fake", true)

        val color = when {
            isReal -> Color.parseColor("#66BB6A")
            isFake -> Color.parseColor("#EF5350")
            else -> Color.parseColor("#FFA726")
        }
        val emoji = when {
            isReal -> "✅"; isFake -> "🚨"; else -> "⚠️"
        }

        panelView?.post {
            resultSection?.visibility = View.VISIBLE

            verdictEmoji?.text = emoji
            verdictLabel?.text = verdict.uppercase()
            verdictLabel?.setTextColor(color)
            confidenceLabel?.text = "${"%.1f".format(confidence * 100)}%"
            confidenceLabel?.setTextColor(color)

            result.videoFakeScore?.let {
                videoScoreText?.text = "Video: ${"%.0f".format(it * 100)}%"
                videoScoreText?.setTextColor(if (it > 0.5) Color.parseColor("#EF5350") else Color.parseColor("#66BB6A"))
                videoScoreText?.visibility = View.VISIBLE
            } ?: run { videoScoreText?.visibility = View.GONE }

            if (isAudioEnabled) {
                result.audioFakeScore?.let {
                    audioScoreText?.text = "Audio: ${"%.0f".format(it * 100)}%"
                    audioScoreText?.setTextColor(if (it > 0.5) Color.parseColor("#EF5350") else Color.parseColor("#66BB6A"))
                } ?: run { audioScoreText?.text = "Audio: N/A" }
                audioScoreText?.visibility = View.VISIBLE
            } else {
                audioScoreText?.text = "Audio: Off"
                audioScoreText?.setTextColor(Color.parseColor("#50FFFFFF"))
                audioScoreText?.visibility = View.VISIBLE
            }

            detailsText?.text = "${result.framesAnalyzed} frames  •  ${"%.0f".format(result.inferenceTimeMs)}ms"

            panelView?.background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE; cornerRadius = dp(16).toFloat()
                setColor(Color.parseColor(if (isReal) "#F0121620" else if (isFake) "#F0181215" else "#F0141820"))
                setStroke(dp(1), Color.parseColor(if (isReal) "#3066BB6A" else if (isFake) "#30EF5350" else "#30FFA726"))
            }
            statusDot?.background = makeDot(color)
            statusText?.text = when {
                isReal -> "📡 ✓ Authentic"
                isFake -> "📡 ✕ Deepfake"
                else -> "📡 ⚠ Suspicious"
            }
            statusText?.setTextColor(color)
            updateHistory(verdict, confidence)
        }
        bubbleView?.post {
            (bubbleView as? TextView)?.apply { text = emoji; setTextColor(color) }
        }
    }

    fun setCapturingState(capturing: Boolean) {
        isCapturing = capturing
        isScanning = false
        panelView?.post {
            startBtn?.apply { isEnabled = !capturing; alpha = if (capturing) 0.4f else 1f }
            stopBtn?.apply { isEnabled = capturing; alpha = if (capturing) 1f else 0.4f }
        }
        if (capturing) {
            cycleNumber = 0; history.clear()
            panelView?.post {
                resultSection?.visibility = View.GONE
                streamSection?.visibility = View.VISIBLE
                historyStrip?.visibility = View.GONE
                streamInfoText?.text = "Connecting…"
                statusDot?.background = makeDot(Color.parseColor("#4FC3F7"))
            }
        } else {
            panelView?.post {
                if (!isScanning) streamSection?.visibility = View.GONE
            }
        }
    }

    // Legacy compat
    fun accumulateResult(r: ApiClient.PredictionResult) = showBatchResult(r)
    fun updateResult(r: ApiClient.PredictionResult) = showBatchResult(r)
    fun updateProgress(e: Int, t: Int, f: Int) {}
    fun updateCycleDuration(s: Int) {}

    // History
    private fun updateHistory(verdict: String, confidence: Float) {
        history.add(CycleResult(verdict, confidence, cycleNumber))
        if (history.size > 5) history.removeAt(0)
        if (history.size <= 1) { historyStrip?.visibility = View.GONE; return }
        historyStrip?.visibility = View.VISIBLE
        val str = history.dropLast(1).joinToString("  ") {
            val e = when (it.verdict) { "Real" -> "✅"; "Suspicious" -> "⚠️"; else -> "🚨" }
            "#${it.num}$e"
        }
        historyStrip?.text = "History: $str"
    }

    // ─── Bubble ────────────────────────────────────────────────

    private fun showBubble() {
        val bubble = TextView(context).apply {
            text = "🛡️"; textSize = 22f; gravity = Gravity.CENTER
            setPadding(dp(6), dp(6), dp(6), dp(6))
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.parseColor("#F0141420"))
                setStroke(dp(1), Color.parseColor("#30FFFFFF"))
            }
            elevation = dp(10).toFloat()
        }
        bubbleParams = WindowManager.LayoutParams(
            dp(52), dp(52),
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply { gravity = Gravity.TOP or Gravity.START; x = dp(8); y = dp(120) }

        var sx = 0; var sy = 0; var tx = 0f; var ty = 0f; var moved = false
        bubble.setOnTouchListener { _, e ->
            when (e.action) {
                MotionEvent.ACTION_DOWN -> { sx = bubbleParams!!.x; sy = bubbleParams!!.y; tx = e.rawX; ty = e.rawY; moved = false; true }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (e.rawX - tx).toInt(); val dy = (e.rawY - ty).toInt()
                    if (Math.abs(dx) > dp(5) || Math.abs(dy) > dp(5)) moved = true
                    bubbleParams!!.x = sx + dx; bubbleParams!!.y = sy + dy
                    windowManager?.updateViewLayout(bubble, bubbleParams); true
                }
                MotionEvent.ACTION_UP -> { if (!moved) toggleExpand(); true }
                else -> false
            }
        }
        bubbleView = bubble
        windowManager?.addView(bubble, bubbleParams)
    }

    private fun removeBubble() {
        bubbleView?.let { try { windowManager?.removeView(it) } catch (_: Exception) {} }
        bubbleView = null
    }

    // ─── Panel ─────────────────────────────────────────────────

    private fun toggleExpand() {
        if (isExpanded) { removePanel(); isExpanded = false }
        else { showPanel(); isExpanded = true }
    }

    private fun showPanel() {
        val panel = buildPanel()
        panelParams = WindowManager.LayoutParams(
            dp(270), WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = (bubbleParams?.x ?: dp(8)) + dp(56)
            y = bubbleParams?.y ?: dp(120)
        }
        panelView = panel
        windowManager?.addView(panel, panelParams)
        setCapturingState(isCapturing)
        if (isScanning) setScanningState(true)
    }

    private fun removePanel() {
        panelView?.let { try { windowManager?.removeView(it) } catch (_: Exception) {} }
        panelView = null; statusText = null; statusDot = null; streamInfoText = null
        verdictEmoji = null; verdictLabel = null; confidenceLabel = null
        videoScoreText = null; audioScoreText = null; detailsText = null
        historyStrip = null; startBtn = null; stopBtn = null
        audioToggleBtn = null; autoToggleBtn = null
        resultSection = null; streamSection = null
    }

    // ─── Build Panel ───────────────────────────────────────────

    private fun buildPanel(): LinearLayout {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE; cornerRadius = dp(16).toFloat()
                setColor(Color.parseColor("#F0121218"))
                setStroke(dp(1), Color.parseColor("#25FFFFFF"))
            }
            elevation = dp(16).toFloat()
        }

        // Header
        val header = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        header.addView(TextView(context).apply {
            text = "🛡️ DeepGuard"; setTextColor(Color.parseColor("#F0F0F0"))
            textSize = 13f; typeface = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(makeHeaderBtn("–") { toggleExpand() })
        header.addView(makeHeaderBtn("✕") { onCloseOverlay?.invoke() })
        root.addView(header)
        root.addView(makeDivider())

        // Status row
        val statusRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, dp(3))
        }
        statusDot = View(context).apply {
            layoutParams = LinearLayout.LayoutParams(dp(7), dp(7)).apply { marginEnd = dp(6) }
            background = makeDot(Color.parseColor("#78909C"))
        }
        statusRow.addView(statusDot)
        statusText = TextView(context).apply {
            text = if (isCapturing) "Streaming…" else if (isScanning) "Scanning…" else "Ready"
            setTextColor(Color.parseColor("#CFD8DC")); textSize = 11f
        }
        statusRow.addView(statusText)
        root.addView(statusRow)

        // Streaming / scanning info
        streamSection = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = if (isCapturing || isScanning) View.VISIBLE else View.GONE
        }
        // Removed duplicated streamInfoText. statusText now handles this completely.
        root.addView(streamSection)

        // Result section
        resultSection = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL; visibility = View.GONE
        }
        val verdictRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, dp(4))
        }
        verdictEmoji = TextView(context).apply { textSize = 28f; setPadding(0, 0, dp(10), 0) }
        verdictRow.addView(verdictEmoji)
        val verdictCol = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        verdictLabel = TextView(context).apply {
            setTextColor(Color.WHITE); textSize = 18f; typeface = Typeface.DEFAULT_BOLD
        }
        verdictCol.addView(verdictLabel)
        confidenceLabel = TextView(context).apply {
            setTextColor(Color.parseColor("#B0FFFFFF")); textSize = 12f
        }
        verdictCol.addView(confidenceLabel)
        verdictRow.addView(verdictCol)
        resultSection!!.addView(verdictRow)

        val scoreRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER
            setPadding(0, dp(4), 0, dp(2))
        }
        videoScoreText = makeScoreChip("Video: —"); scoreRow.addView(videoScoreText)
        audioScoreText = makeScoreChip("Audio: Off"); scoreRow.addView(audioScoreText)
        resultSection!!.addView(scoreRow)

        detailsText = TextView(context).apply {
            setTextColor(Color.parseColor("#40FFFFFF")); textSize = 9f
            setPadding(0, dp(2), 0, dp(2))
        }
        resultSection!!.addView(detailsText)
        root.addView(resultSection)

        historyStrip = TextView(context).apply {
            setTextColor(Color.parseColor("#40FFFFFF")); textSize = 9f
            setPadding(0, dp(2), 0, dp(2)); visibility = View.GONE
        }
        root.addView(historyStrip)
        root.addView(makeDivider())

        // ── Toggles row: Auto + Audio ──
        val togglesRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(4), 0, dp(4))
        }

        // Auto toggle
        autoToggleBtn = TextView(context).apply {
            text = if (isAutoEnabled) "AUTO-SCAN: ON" else "AUTO-SCAN: OFF"
            setTextColor(if (isAutoEnabled) Color.parseColor("#4FC3F7") else Color.parseColor("#78909C"))
            textSize = 9f; typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER
            setPadding(dp(8), dp(6), dp(8), dp(6))
            background = makeToggleBg(isAutoEnabled, "#4FC3F7")
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) }
            setOnClickListener {
                isAutoEnabled = !isAutoEnabled
                prefs.edit().putBoolean(KEY_AUTO_MODE, isAutoEnabled).apply()
                text = if (isAutoEnabled) "AUTO-SCAN: ON" else "AUTO-SCAN: OFF"
                setTextColor(if (isAutoEnabled) Color.parseColor("#4FC3F7") else Color.parseColor("#78909C"))
                background = makeToggleBg(isAutoEnabled, "#4FC3F7")
                onAutoToggleChanged?.invoke(isAutoEnabled)
            }
        }
        togglesRow.addView(autoToggleBtn)

        // Audio toggle
        audioToggleBtn = TextView(context).apply {
            text = if (isAudioEnabled) "AUDIO: ON" else "AUDIO: OFF"
            setTextColor(if (isAudioEnabled) Color.parseColor("#66BB6A") else Color.parseColor("#78909C"))
            textSize = 9f; typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER
            setPadding(dp(8), dp(6), dp(8), dp(6))
            background = makeToggleBg(isAudioEnabled, "#66BB6A")
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener {
                isAudioEnabled = !isAudioEnabled
                prefs.edit().putBoolean(KEY_AUDIO_ENABLED, isAudioEnabled).apply()
                text = if (isAudioEnabled) "AUDIO: ON" else "AUDIO: OFF"
                setTextColor(if (isAudioEnabled) Color.parseColor("#66BB6A") else Color.parseColor("#78909C"))
                background = makeToggleBg(isAudioEnabled, "#66BB6A")
                onAudioToggleChanged?.invoke(isAudioEnabled)
            }
        }
        togglesRow.addView(audioToggleBtn)
        root.addView(togglesRow)
        root.addView(makeDivider())

        // Buttons
        val btnRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(6), 0, 0); gravity = Gravity.CENTER
        }
        startBtn = makeActionBtn("START", "#2E7D32") {
            val url = prefs.getString("server_url", "") ?: ""
            if (url.isNotBlank()) onStartCapture?.invoke(url)
            else updateStatus("⚠ Set URL in app")
        }
        startBtn!!.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(4) }
        btnRow.addView(startBtn)

        stopBtn = makeActionBtn("STOP", "#C62828") { onStopCapture?.invoke() }
        stopBtn!!.alpha = 0.4f; stopBtn!!.isEnabled = false
        stopBtn!!.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(4) }
        btnRow.addView(stopBtn)
        root.addView(btnRow)

        return root
    }

    // ─── Helpers ───────────────────────────────────────────────

    private fun makeHeaderBtn(text: String, onClick: () -> Unit) = TextView(context).apply {
        this.text = text; setTextColor(Color.parseColor("#80FFFFFF")); textSize = 13f
        typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER
        background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE; cornerRadius = dp(6).toFloat()
            setColor(Color.parseColor("#10FFFFFF"))
        }
        setPadding(dp(8), dp(2), dp(8), dp(2))
        layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
        ).apply { marginStart = dp(4) }
        setOnClickListener { onClick() }
    }

    private fun makeActionBtn(text: String, c: String, onClick: () -> Unit) = TextView(context).apply {
        this.text = text; setTextColor(Color.WHITE); textSize = 11f
        typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER
        setPadding(dp(10), dp(8), dp(10), dp(8))
        background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE; cornerRadius = dp(10).toFloat()
            setColor(Color.parseColor(c))
        }
        setOnClickListener { onClick() }
    }

    private fun makeScoreChip(label: String) = TextView(context).apply {
        text = label; setTextColor(Color.parseColor("#B0FFFFFF")); textSize = 10f
        typeface = Typeface.DEFAULT_BOLD; gravity = Gravity.CENTER
        setPadding(dp(10), dp(5), dp(10), dp(5))
        background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE; cornerRadius = dp(8).toFloat()
            setColor(Color.parseColor("#10FFFFFF"))
        }
        layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginStart = dp(2); marginEnd = dp(2)
        }
    }

    private fun makeToggleBg(on: Boolean, accentHex: String) = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE; cornerRadius = dp(8).toFloat()
        setColor(if (on) Color.parseColor("#1A${accentHex.removePrefix("#")}") else Color.parseColor("#10FFFFFF"))
        setStroke(1, if (on) Color.parseColor("#30${accentHex.removePrefix("#")}") else Color.parseColor("#15FFFFFF"))
    }

    private fun makeDot(color: Int) = GradientDrawable().apply {
        shape = GradientDrawable.OVAL; setColor(color); setSize(dp(7), dp(7))
    }

    private fun makeDivider() = View(context).apply {
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1).apply {
            topMargin = dp(4); bottomMargin = dp(2)
        }
        setBackgroundColor(Color.parseColor("#15FFFFFF"))
    }
}
