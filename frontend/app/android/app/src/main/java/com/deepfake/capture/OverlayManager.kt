package com.deepfake.capture

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * Floating overlay with two states:
 *   • Collapsed: small draggable bubble icon
 *   • Expanded: panel with cycle-based detection summary, server URL, start/stop controls
 *
 * Accumulates per-frame results over 30-second cycles, then displays a structured
 * briefing template with verdict, confidence, frame breakdown, alerts, and cycle history.
 */
class OverlayManager(private val context: Context) {

    companion object {
        private const val TAG = "OverlayManager"
        private const val PREFS_NAME = "deepfake_prefs"
        private const val KEY_SERVER_URL = "server_url"
        private const val CYCLE_FRAME_COUNT = 30          // Frames per cycle
        private const val CYCLE_DURATION_MS = 30_000L     // Max cycle duration in ms
        private const val MAX_HISTORY = 5                 // Keep last N cycle summaries
    }

    private var windowManager: WindowManager? = null

    // Collapsed bubble
    private var bubbleView: View? = null
    private var bubbleParams: WindowManager.LayoutParams? = null

    // Expanded panel
    private var panelView: View? = null
    private var panelParams: WindowManager.LayoutParams? = null

    // Panel UI references
    private var statusText: TextView? = null
    private var progressText: TextView? = null
    private var progressBar: View? = null
    private var progressBarTrack: View? = null
    private var verdictEmoji: TextView? = null
    private var verdictLabel: TextView? = null
    private var confidenceLabel: TextView? = null
    private var breakdownCard: LinearLayout? = null
    private var breakdownText: TextView? = null
    private var scoresLine: TextView? = null
    private var alertSection: TextView? = null
    private var inferenceLabel: TextView? = null
    private var historyStrip: TextView? = null
    private var cycleCountLabel: TextView? = null
    private var serverUrlInput: EditText? = null
    private var startBtn: TextView? = null
    private var stopBtn: TextView? = null
    private var statusDot: View? = null

    // Sections that should hide/show
    private var resultSection: LinearLayout? = null

    private var isExpanded = false
    private var isCapturing = false

    var onStartCapture: ((serverUrl: String) -> Unit)? = null
    var onStopCapture: (() -> Unit)? = null
    var onCloseOverlay: (() -> Unit)? = null

    // ─── Cycle Accumulation ────────────────────────────────────

    data class CycleSummary(
        val cycleNumber: Int,
        val verdict: String,           // "Real", "Fake", "Suspicious"
        val avgConfidence: Float,
        val realCount: Int,
        val fakeCount: Int,
        val suspiciousCount: Int,
        val totalFrames: Int,
        val avgVideoScore: Float?,
        val avgAudioScore: Float?,
        val alerts: List<String>,
        val avgInferenceMs: Float,
        val timestamp: Long
    )

    private val frameResults = mutableListOf<ApiClient.PredictionResult>()
    private var cycleStartTime = 0L
    private var currentCycleNumber = 0
    private val cycleHistory = mutableListOf<CycleSummary>()
    private var latestSummary: CycleSummary? = null

    // ─── dp/sp helper ──────────────────────────────────────────
    private fun dp(v: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), context.resources.displayMetrics).toInt()

    private fun sp(v: Float): Float =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, context.resources.displayMetrics)

    // ─── Public API ────────────────────────────────────────────

    fun show() {
        if (bubbleView != null) return

        windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        showBubble()
        Log.i(TAG, "Overlay shown (bubble)")
    }

    fun hide() {
        removeBubble()
        removePanel()
        windowManager = null
        Log.i(TAG, "Overlay hidden")
    }

    fun updateStatus(status: String) {
        panelView?.post {
            statusText?.text = status
            statusDot?.background = makeDot(Color.parseColor("#4FC3F7"))
        }
        bubbleView?.post {
            (bubbleView as? TextView)?.setTextColor(Color.parseColor("#4FC3F7"))
        }
    }

    /**
     * Accumulate a single per-frame result into the current cycle.
     * After CYCLE_FRAME_COUNT frames or CYCLE_DURATION_MS ms, finalize
     * the cycle and display the briefing template.
     */
    fun accumulateResult(result: ApiClient.PredictionResult) {
        if (cycleStartTime == 0L) {
            cycleStartTime = System.currentTimeMillis()
        }

        frameResults.add(result)

        val elapsed = System.currentTimeMillis() - cycleStartTime
        val frameCount = frameResults.size

        // Update live progress
        panelView?.post {
            progressText?.visibility = View.VISIBLE
            progressBarTrack?.visibility = View.VISIBLE
            updateProgressUI(frameCount, CYCLE_FRAME_COUNT)
            statusText?.text = "Analyzing…"
            statusDot?.background = makeDot(Color.parseColor("#4FC3F7"))
        }

        // Check if cycle is complete
        if (frameCount >= CYCLE_FRAME_COUNT || elapsed >= CYCLE_DURATION_MS) {
            finalizeCycle()
        }
    }

    /**
     * Legacy method — calls accumulateResult internally.
     */
    fun updateResult(result: ApiClient.PredictionResult) {
        accumulateResult(result)
    }

    /**
     * Legacy method for backward compat.
     */
    fun updateResult(prediction: String, confidence: Float) {
        val result = ApiClient.PredictionResult(
            prediction = prediction,
            confidence = confidence,
            fakeProbability = if (prediction.equals("Fake", ignoreCase = true)) confidence else 1f - confidence,
            videoFakeScore = null,
            audioFakeScore = null,
            securityAlert = null,
            threatVector = null,
            analysisSource = "local",
            fusionMode = null,
            videoWeight = null,
            audioWeight = null,
            inferenceTimeMs = 0f
        )
        accumulateResult(result)
    }

    fun setCapturingState(capturing: Boolean) {
        isCapturing = capturing
        panelView?.post {
            startBtn?.apply {
                isEnabled = !capturing
                alpha = if (capturing) 0.4f else 1.0f
            }
            stopBtn?.apply {
                isEnabled = capturing
                alpha = if (capturing) 1.0f else 0.4f
            }
            serverUrlInput?.isEnabled = !capturing
        }

        if (capturing) {
            // Reset accumulator for new session
            resetAccumulator()
            panelView?.post {
                progressText?.visibility = View.VISIBLE
                progressBarTrack?.visibility = View.VISIBLE
                updateProgressUI(0, CYCLE_FRAME_COUNT)
                showResultSection(false)
                historyStrip?.visibility = View.GONE
            }
        } else {
            panelView?.post {
                progressText?.visibility = View.GONE
                progressBarTrack?.visibility = View.GONE
            }
        }
    }

    // ─── Cycle Logic ───────────────────────────────────────────

    private fun resetAccumulator() {
        frameResults.clear()
        cycleStartTime = 0L
        currentCycleNumber = 0
        cycleHistory.clear()
        latestSummary = null
    }

    private fun finalizeCycle() {
        if (frameResults.isEmpty()) return

        currentCycleNumber++

        // Count verdicts
        var realCount = 0
        var fakeCount = 0
        var suspiciousCount = 0
        var totalConfidence = 0f
        var totalVideoScore = 0f
        var videoScoreCount = 0
        var totalAudioScore = 0f
        var audioScoreCount = 0
        var totalInferenceMs = 0f
        val uniqueAlerts = mutableSetOf<String>()

        for (r in frameResults) {
            totalConfidence += r.confidence
            totalInferenceMs += r.inferenceTimeMs

            when {
                r.prediction.equals("Real", ignoreCase = true) -> realCount++
                r.prediction.equals("Suspicious", ignoreCase = true) -> suspiciousCount++
                else -> fakeCount++
            }

            r.videoFakeScore?.let {
                totalVideoScore += it
                videoScoreCount++
            }
            r.audioFakeScore?.let {
                totalAudioScore += it
                audioScoreCount++
            }
            r.securityAlert?.let { uniqueAlerts.add(it) }
            r.threatVector?.let { uniqueAlerts.add(it) }
        }

        val totalFrames = frameResults.size

        // Majority vote for verdict
        val verdict = when {
            fakeCount >= realCount && fakeCount >= suspiciousCount -> "Fake"
            suspiciousCount > realCount -> "Suspicious"
            else -> "Real"
        }

        val summary = CycleSummary(
            cycleNumber = currentCycleNumber,
            verdict = verdict,
            avgConfidence = totalConfidence / totalFrames,
            realCount = realCount,
            fakeCount = fakeCount,
            suspiciousCount = suspiciousCount,
            totalFrames = totalFrames,
            avgVideoScore = if (videoScoreCount > 0) totalVideoScore / videoScoreCount else null,
            avgAudioScore = if (audioScoreCount > 0) totalAudioScore / audioScoreCount else null,
            alerts = uniqueAlerts.toList(),
            avgInferenceMs = totalInferenceMs / totalFrames,
            timestamp = System.currentTimeMillis()
        )

        latestSummary = summary
        cycleHistory.add(summary)
        if (cycleHistory.size > MAX_HISTORY) {
            cycleHistory.removeAt(0)
        }

        // Reset for next cycle
        frameResults.clear()
        cycleStartTime = System.currentTimeMillis()

        // Update UI
        panelView?.post {
            showCycleBriefing(summary)
        }

        // Update bubble
        val emoji = when (verdict) {
            "Real" -> "✅"
            "Suspicious" -> "⚠️"
            else -> "🚨"
        }
        val color = when (verdict) {
            "Real" -> Color.parseColor("#66BB6A")
            "Suspicious" -> Color.parseColor("#FFA726")
            else -> Color.parseColor("#EF5350")
        }
        bubbleView?.post {
            (bubbleView as? TextView)?.apply {
                text = emoji
                setTextColor(color)
            }
        }

        Log.i(TAG, "Cycle #$currentCycleNumber finalized: $verdict " +
                "(R=$realCount F=$fakeCount S=$suspiciousCount, " +
                "conf=${"%.1f".format(summary.avgConfidence * 100)}%)")
    }

    // ─── UI Updates ────────────────────────────────────────────

    private fun updateProgressUI(current: Int, total: Int) {
        progressText?.text = "Analyzing: $current / $total frames"

        // Update progress bar width
        progressBar?.let { bar ->
            progressBarTrack?.let { track ->
                track.post {
                    val trackWidth = track.width
                    if (trackWidth > 0) {
                        val fillWidth = (trackWidth * current.coerceAtMost(total)) / total
                        val params = bar.layoutParams
                        params.width = fillWidth.coerceAtLeast(0)
                        bar.layoutParams = params
                    }
                }
            }
        }
    }

    private fun showResultSection(visible: Boolean) {
        resultSection?.visibility = if (visible) View.VISIBLE else View.GONE
    }

    private fun showCycleBriefing(summary: CycleSummary) {
        showResultSection(true)

        val isReal = summary.verdict == "Real"
        val isSuspicious = summary.verdict == "Suspicious"

        val color = when {
            isReal -> Color.parseColor("#66BB6A")
            isSuspicious -> Color.parseColor("#FFA726")
            else -> Color.parseColor("#EF5350")
        }

        val emoji = when {
            isReal -> "✅"
            isSuspicious -> "⚠️"
            else -> "🚨"
        }

        // Cycle label
        cycleCountLabel?.text = "CYCLE RESULT — #${summary.cycleNumber}"
        cycleCountLabel?.setTextColor(Color.parseColor("#80FFFFFF"))

        // Verdict
        verdictEmoji?.text = emoji
        verdictLabel?.text = "${summary.verdict.uppercase()}"
        verdictLabel?.setTextColor(color)

        // Confidence
        confidenceLabel?.text = "${"%.1f".format(summary.avgConfidence * 100)}% confidence"
        confidenceLabel?.setTextColor(color)

        // Frame breakdown
        breakdownText?.text = "Real: ${summary.realCount}   Fake: ${summary.fakeCount}   Suspicious: ${summary.suspiciousCount}"

        // Scores line
        val scoreParts = mutableListOf<String>()
        summary.avgVideoScore?.let { scoreParts.add("Avg Vid: ${"%.0f".format(it * 100)}%") }
        summary.avgAudioScore?.let { scoreParts.add("Avg Aud: ${"%.0f".format(it * 100)}%") }
        scoresLine?.text = if (scoreParts.isNotEmpty()) scoreParts.joinToString("   ") else "Scores: N/A"

        // Alerts
        if (summary.alerts.isEmpty()) {
            alertSection?.text = "✓ No alerts detected"
            alertSection?.setTextColor(Color.parseColor("#66BB6A"))
        } else {
            val alertStr = summary.alerts.joinToString("\n") { alert ->
                when (alert) {
                    "FACE_SWAP_ATTACK" -> "⚠ Face swap detected"
                    "VOICE_CLONE_ATTACK" -> "⚠ Voice clone detected"
                    "MISMATCH_DETECTED" -> "⚠ Audio/video mismatch"
                    "LOW_CONFIDENCE" -> "⚠ Low confidence"
                    else -> "⚠ $alert"
                }
            }
            alertSection?.text = alertStr
            alertSection?.setTextColor(Color.parseColor("#FFA726"))
        }

        // Inference time
        inferenceLabel?.text = "${"%.0f".format(summary.avgInferenceMs)}ms avg inference"

        // Subtle panel background tint
        panelView?.background = makeRoundedRect(
            when {
                isReal -> "#0A66BB6A"
                isSuspicious -> "#0AFFA726"
                else -> "#0AEF5350"
            }, 20
        )

        statusDot?.background = makeDot(color)
        statusText?.text = when {
            isReal -> "Real Detected"
            isSuspicious -> "Suspicious!"
            else -> "Fake Detected"
        }
        statusText?.setTextColor(color)

        // History strip
        updateHistoryStrip()

        // Reset progress for next cycle
        updateProgressUI(0, CYCLE_FRAME_COUNT)
    }

    private fun updateHistoryStrip() {
        if (cycleHistory.size <= 1) {
            historyStrip?.visibility = View.GONE
            return
        }

        historyStrip?.visibility = View.VISIBLE
        // Show all but the latest (which is the current one displayed)
        val pastCycles = cycleHistory.dropLast(1).takeLast(4)
        val historyStr = pastCycles.joinToString("   ") { cycle ->
            val e = when (cycle.verdict) {
                "Real" -> "✅"
                "Suspicious" -> "⚠️"
                else -> "🚨"
            }
            "#${cycle.cycleNumber} $e ${cycle.verdict}"
        }
        historyStrip?.text = "Past: $historyStr"
    }

    // ─── Collapsed Bubble ──────────────────────────────────────

    private fun showBubble() {
        val bubble = TextView(context).apply {
            text = "🛡️"
            textSize = 24f
            gravity = Gravity.CENTER
            setPadding(dp(4), dp(4), dp(4), dp(4))
            background = makeRoundedRect("#E6181825", 50)
            elevation = dp(8).toFloat()
        }

        bubbleParams = WindowManager.LayoutParams(
            dp(52), dp(52),
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = dp(8)
            y = dp(120)
        }

        // Drag + tap detection
        var startX = 0
        var startY = 0
        var startTouchX = 0f
        var startTouchY = 0f
        var moved = false

        bubble.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    startX = bubbleParams!!.x
                    startY = bubbleParams!!.y
                    startTouchX = event.rawX
                    startTouchY = event.rawY
                    moved = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - startTouchX).toInt()
                    val dy = (event.rawY - startTouchY).toInt()
                    if (Math.abs(dx) > dp(5) || Math.abs(dy) > dp(5)) moved = true
                    bubbleParams!!.x = startX + dx
                    bubbleParams!!.y = startY + dy
                    windowManager?.updateViewLayout(bubble, bubbleParams)
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!moved) toggleExpand()
                    true
                }
                else -> false
            }
        }

        bubbleView = bubble
        windowManager?.addView(bubble, bubbleParams)
    }

    private fun removeBubble() {
        bubbleView?.let {
            try { windowManager?.removeView(it) } catch (_: Exception) {}
        }
        bubbleView = null
    }

    // ─── Expanded Panel ────────────────────────────────────────

    private fun toggleExpand() {
        if (isExpanded) {
            removePanel()
            isExpanded = false
        } else {
            showPanel()
            isExpanded = true
        }
    }

    private fun showPanel() {
        val panel = buildPanelLayout()

        panelParams = WindowManager.LayoutParams(
            dp(310),
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = (bubbleParams?.x ?: dp(8)) + dp(56)
            y = bubbleParams?.y ?: dp(120)
            softInputMode = WindowManager.LayoutParams.SOFT_INPUT_ADJUST_PAN
        }

        panelView = panel
        windowManager?.addView(panel, panelParams)

        // Refresh button state
        setCapturingState(isCapturing)

        // If we have a previous summary, show it
        latestSummary?.let { showCycleBriefing(it) }
        if (isCapturing) {
            updateProgressUI(frameResults.size, CYCLE_FRAME_COUNT)
        }
    }

    private fun removePanel() {
        panelView?.let {
            try { windowManager?.removeView(it) } catch (_: Exception) {}
        }
        panelView = null
        statusText = null
        progressText = null
        progressBar = null
        progressBarTrack = null
        verdictEmoji = null
        verdictLabel = null
        confidenceLabel = null
        breakdownCard = null
        breakdownText = null
        scoresLine = null
        alertSection = null
        inferenceLabel = null
        historyStrip = null
        cycleCountLabel = null
        resultSection = null
        serverUrlInput = null
        startBtn = null
        stopBtn = null
        statusDot = null
    }

    private fun buildPanelLayout(): LinearLayout {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val savedUrl = prefs.getString(KEY_SERVER_URL, "http://10.0.2.2:7860") ?: ""

        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(14))
            background = makeRoundedRect("#E6181825", 20)
            elevation = dp(12).toFloat()
        }

        // ── Header row: title + close button
        val header = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val title = TextView(context).apply {
            text = "🛡️ DeepFake Detector"
            setTextColor(Color.WHITE)
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        header.addView(title)

        val minimizeBtn = TextView(context).apply {
            text = "—"
            setTextColor(Color.parseColor("#80FFFFFF"))
            textSize = 16f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(dp(8), 0, dp(8), 0)
            setOnClickListener { toggleExpand() }
        }
        header.addView(minimizeBtn)

        val closeBtn = TextView(context).apply {
            text = "✕"
            setTextColor(Color.parseColor("#80FFFFFF"))
            textSize = 18f
            gravity = Gravity.CENTER
            setPadding(dp(8), 0, 0, 0)
            setOnClickListener { onCloseOverlay?.invoke() }
        }
        header.addView(closeBtn)
        root.addView(header)

        // ── Divider
        root.addView(makeDivider())

        // ── Status row
        val statusRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(6), 0, dp(2))
        }

        statusDot = View(context).apply {
            layoutParams = LinearLayout.LayoutParams(dp(8), dp(8)).apply {
                marginEnd = dp(8)
            }
            background = makeDot(Color.parseColor("#78909C"))
        }
        statusRow.addView(statusDot)

        statusText = TextView(context).apply {
            text = if (isCapturing) "Capturing…" else "Idle"
            setTextColor(Color.parseColor("#B0BEC5"))
            textSize = 13f
        }
        statusRow.addView(statusText)
        root.addView(statusRow)

        // ── Progress section (frame count + bar)
        progressText = TextView(context).apply {
            text = "Analyzing: 0 / $CYCLE_FRAME_COUNT frames"
            setTextColor(Color.parseColor("#80FFFFFF"))
            textSize = 11f
            setPadding(0, dp(6), 0, dp(4))
            visibility = if (isCapturing) View.VISIBLE else View.GONE
        }
        root.addView(progressText)

        // Progress bar track
        progressBarTrack = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(4)
            ).apply {
                bottomMargin = dp(6)
            }
            background = makeRoundedRect("#20FFFFFF", 4)
            visibility = if (isCapturing) View.VISIBLE else View.GONE
        }

        progressBar = View(context).apply {
            layoutParams = LinearLayout.LayoutParams(0, dp(4))
            background = makeRoundedRect("#4FC3F7", 4)
        }
        (progressBarTrack as LinearLayout).addView(progressBar)
        root.addView(progressBarTrack)

        // ── Result section (hidden until first cycle completes)
        resultSection = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }

        // Cycle label
        cycleCountLabel = TextView(context).apply {
            text = "CYCLE RESULT"
            setTextColor(Color.parseColor("#80FFFFFF"))
            textSize = 10f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, dp(4), 0, dp(4))
            setLetterSpacing(0.1f)
        }
        resultSection!!.addView(cycleCountLabel)

        // Verdict row (emoji + label)
        val verdictRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(2), 0, 0)
        }

        verdictEmoji = TextView(context).apply {
            text = ""
            textSize = 28f
            setPadding(0, 0, dp(10), 0)
        }
        verdictRow.addView(verdictEmoji)

        val verdictCol = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
        }

        verdictLabel = TextView(context).apply {
            text = ""
            setTextColor(Color.WHITE)
            textSize = 22f
            typeface = Typeface.DEFAULT_BOLD
        }
        verdictCol.addView(verdictLabel)

        confidenceLabel = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#B0FFFFFF"))
            textSize = 13f
        }
        verdictCol.addView(confidenceLabel)

        verdictRow.addView(verdictCol)
        resultSection!!.addView(verdictRow)

        // Frame breakdown card
        breakdownCard = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(8), dp(12), dp(8))
            background = makeRoundedRect("#15FFFFFF", 10)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = dp(8)
                bottomMargin = dp(4)
            }
        }

        val breakdownLabel = TextView(context).apply {
            text = "Frame Breakdown"
            setTextColor(Color.parseColor("#60FFFFFF"))
            textSize = 10f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, 0, 0, dp(4))
            setLetterSpacing(0.08f)
        }
        breakdownCard!!.addView(breakdownLabel)

        breakdownText = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#E0FFFFFF"))
            textSize = 12f
        }
        breakdownCard!!.addView(breakdownText)

        scoresLine = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#80FFFFFF"))
            textSize = 11f
            setPadding(0, dp(3), 0, 0)
        }
        breakdownCard!!.addView(scoresLine)

        resultSection!!.addView(breakdownCard)

        // Alert section
        alertSection = TextView(context).apply {
            text = ""
            textSize = 12f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(0, dp(4), 0, dp(2))
        }
        resultSection!!.addView(alertSection)

        // Inference time
        inferenceLabel = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#60FFFFFF"))
            textSize = 10f
            setPadding(0, dp(2), 0, dp(4))
        }
        resultSection!!.addView(inferenceLabel)

        root.addView(resultSection)

        // ── History strip divider + text
        root.addView(makeDivider())

        historyStrip = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#60FFFFFF"))
            textSize = 10f
            setPadding(0, dp(4), 0, dp(2))
            visibility = View.GONE
        }
        root.addView(historyStrip)

        // ── Divider before server
        root.addView(makeDivider())

        // ── Server URL
        val urlLabel = TextView(context).apply {
            text = "Server URL"
            setTextColor(Color.parseColor("#60FFFFFF"))
            textSize = 11f
            setPadding(0, dp(6), 0, dp(2))
        }
        root.addView(urlLabel)

        serverUrlInput = EditText(context).apply {
            setText(savedUrl)
            setTextColor(Color.WHITE)
            textSize = 12f
            setHintTextColor(Color.parseColor("#40FFFFFF"))
            hint = "http://10.0.2.2:7860"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            imeOptions = EditorInfo.IME_ACTION_DONE
            setPadding(dp(10), dp(6), dp(10), dp(6))
            background = makeRoundedRect("#20FFFFFF", 8)
            isSingleLine = true
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }
        root.addView(serverUrlInput)

        // ── Buttons row
        val btnRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(10), 0, 0)
            gravity = Gravity.CENTER
        }

        startBtn = TextView(context).apply {
            text = "▶  Start"
            setTextColor(Color.WHITE)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(10), dp(16), dp(10))
            background = makeRoundedRect("#4CAF50", 12)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = dp(6)
            }
            setOnClickListener {
                val url = serverUrlInput?.text?.toString()?.trim() ?: ""
                if (url.isNotBlank()) {
                    context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                        .edit().putString(KEY_SERVER_URL, url).apply()
                    onStartCapture?.invoke(url)
                }
            }
        }
        btnRow.addView(startBtn)

        stopBtn = TextView(context).apply {
            text = "⏹  Stop"
            setTextColor(Color.WHITE)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(10), dp(16), dp(10))
            background = makeRoundedRect("#EF5350", 12)
            alpha = 0.4f
            isEnabled = false
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginStart = dp(6)
            }
            setOnClickListener {
                onStopCapture?.invoke()
            }
        }
        btnRow.addView(stopBtn)
        root.addView(btnRow)

        return root
    }

    // ─── Drawable Helpers ──────────────────────────────────────

    private fun makeRoundedRect(colorHex: String, radiusDp: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = dp(radiusDp).toFloat()
            setColor(Color.parseColor(colorHex))
            setStroke(1, Color.parseColor("#20FFFFFF"))
        }
    }

    private fun makeDot(color: Int): GradientDrawable {
        return GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(color)
            setSize(dp(8), dp(8))
        }
    }

    private fun makeDivider(): View {
        return View(context).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 1
            ).apply {
                topMargin = dp(6)
                bottomMargin = dp(2)
            }
            setBackgroundColor(Color.parseColor("#15FFFFFF"))
        }
    }
}
