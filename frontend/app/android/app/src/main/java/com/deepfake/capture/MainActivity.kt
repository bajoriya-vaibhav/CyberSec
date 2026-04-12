package com.deepfake.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import org.json.JSONObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.InputStream
import java.util.concurrent.TimeUnit

/**
 * Main screen for DeepGuard.
 *
 * Two independent features:
 *   1. Real-time overlay → screen capture + WebSocket streaming (unchanged)
 *   2. File analysis → pick image/video → POST /analyze_file → show result
 *      + optional "Mark as Real/Fake" → POST /verify (saves to server dataset)
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val OVERLAY_PERMISSION_REQUEST = 1001
        private const val PREFS_NAME = "deepfake_prefs"
        private const val KEY_SERVER_URL = "server_url"
        private const val KEY_ANALYSIS_DURATION = "analysis_duration"
    }

    private lateinit var projectionLauncher: ActivityResultLauncher<Intent>
    private lateinit var permissionLauncher: ActivityResultLauncher<Array<String>>
    private lateinit var imagePickerLauncher: ActivityResultLauncher<String>
    private lateinit var videoPickerLauncher: ActivityResultLauncher<String>
    private lateinit var prefs: SharedPreferences

    // Settings UI
    private lateinit var serverUrlInput: EditText
    private lateinit var durationSeekBar: SeekBar
    private lateinit var durationValueText: TextView

    // File analysis UI
    private lateinit var txtSelectedFile: TextView
    private lateinit var btnAnalyseFile: Button
    private lateinit var txtAnalysisStatus: TextView
    private lateinit var resultCard: LinearLayout
    private lateinit var txtVerdictEmoji: TextView
    private lateinit var txtVerdictLabel: TextView
    private lateinit var txtConfidence: TextView
    private lateinit var txtResultDetails: TextView
    private lateinit var btnMarkReal: Button
    private lateinit var btnMarkFake: Button
    private lateinit var txtVerifyStatus: TextView

    // State
    private var selectedFileUri: Uri? = null
    private var selectedFileName: String = ""
    private var selectedMimeType: String = ""
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS) // Video analysis can take a while
        .build()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)

        // ── Settings UI ────────────────────────────────────────
        serverUrlInput = findViewById(R.id.editServerUrl)
        durationSeekBar = findViewById(R.id.seekDuration)
        durationValueText = findViewById(R.id.txtDurationValue)

        val savedUrl = prefs.getString(KEY_SERVER_URL, "http://10.0.2.2:7860") ?: ""
        serverUrlInput.setText(savedUrl)

        val savedDuration = prefs.getInt(KEY_ANALYSIS_DURATION, 15)
        durationSeekBar.progress = savedDuration
        durationValueText.text = "$savedDuration seconds"

        durationSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress.coerceAtLeast(3)
                durationValueText.text = "$value seconds"
                prefs.edit().putInt(KEY_ANALYSIS_DURATION, value).apply()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        serverUrlInput.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) saveServerUrl()
        }

        // ── File Analysis UI ───────────────────────────────────
        txtSelectedFile = findViewById(R.id.txtSelectedFile)
        btnAnalyseFile = findViewById(R.id.btnAnalyseFile)
        txtAnalysisStatus = findViewById(R.id.txtAnalysisStatus)
        resultCard = findViewById(R.id.resultCard)
        txtVerdictEmoji = findViewById(R.id.txtVerdictEmoji)
        txtVerdictLabel = findViewById(R.id.txtVerdictLabel)
        txtConfidence = findViewById(R.id.txtConfidence)
        txtResultDetails = findViewById(R.id.txtResultDetails)
        btnMarkReal = findViewById(R.id.btnMarkReal)
        btnMarkFake = findViewById(R.id.btnMarkFake)
        txtVerifyStatus = findViewById(R.id.txtVerifyStatus)

        // ── Register launchers ─────────────────────────────────
        projectionLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                saveServerUrl()
                launchServiceWithOverlay(result.resultCode, result.data!!)
            } else {
                Toast.makeText(this, "Screen capture permission denied", Toast.LENGTH_SHORT).show()
            }
        }

        permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { permissions ->
            if (permissions.values.all { it }) {
                checkOverlayPermission()
            } else {
                Toast.makeText(this, "Audio permission required", Toast.LENGTH_SHORT).show()
            }
        }

        imagePickerLauncher = registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->
            uri?.let { onFileSelected(it, isVideo = false) }
        }

        videoPickerLauncher = registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri ->
            uri?.let { onFileSelected(it, isVideo = true) }
        }

        // ── Button clicks ──────────────────────────────────────
        findViewById<Button>(R.id.btnStartOverlay).setOnClickListener {
            saveServerUrl()
            val url = serverUrlInput.text.toString().trim()
            if (url.isBlank()) {
                Toast.makeText(this, "Please enter a server URL", Toast.LENGTH_SHORT).show()
                serverUrlInput.requestFocus()
                return@setOnClickListener
            }
            startPermissionFlow()
        }

        findViewById<Button>(R.id.btnPickImage).setOnClickListener {
            imagePickerLauncher.launch("image/*")
        }

        findViewById<Button>(R.id.btnPickVideo).setOnClickListener {
            videoPickerLauncher.launch("video/*")
        }

        btnAnalyseFile.setOnClickListener {
            val uri = selectedFileUri ?: return@setOnClickListener
            val serverUrl = serverUrlInput.text.toString().trim()
            if (serverUrl.isBlank()) {
                Toast.makeText(this, "Enter server URL first", Toast.LENGTH_SHORT).show()
                serverUrlInput.requestFocus()
                return@setOnClickListener
            }
            analyseFile(uri, serverUrl)
        }

        btnMarkReal.setOnClickListener { verifyFile("real") }
        btnMarkFake.setOnClickListener { verifyFile("fake") }
    }

    // ─── File Selection ────────────────────────────────────────

    private fun onFileSelected(uri: Uri, isVideo: Boolean) {
        selectedFileUri = uri
        selectedMimeType = if (isVideo) "video/mp4" else "image/jpeg"

        // Resolve display name
        var name = uri.lastPathSegment ?: "selected_file"
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val col = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && col >= 0) name = cursor.getString(col)
        }
        selectedFileName = name
        txtSelectedFile.text = name

        // Enable analyse button
        btnAnalyseFile.isEnabled = true
        btnAnalyseFile.alpha = 1f
        btnAnalyseFile.backgroundTintList = android.content.res.ColorStateList.valueOf(
            Color.parseColor("#1565C0")
        )
        btnAnalyseFile.setTextColor(Color.WHITE)

        // Reset result card
        resultCard.visibility = View.GONE
        txtVerifyStatus.visibility = View.GONE
    }

    // ─── File Analysis ─────────────────────────────────────────

    private fun analyseFile(uri: Uri, serverUrl: String) {
        scope.launch {
            // Lock UI
            btnAnalyseFile.isEnabled = false
            txtAnalysisStatus.text = "⏳ Uploading and analysing…"
            txtAnalysisStatus.visibility = View.VISIBLE
            resultCard.visibility = View.GONE
            txtVerifyStatus.visibility = View.GONE

            try {
                val bytes = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use(InputStream::readBytes)
                } ?: throw Exception("Could not read file")

                val response = withContext(Dispatchers.IO) {
                    val requestBody = MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart(
                            "file", selectedFileName,
                            bytes.toRequestBody(selectedMimeType.toMediaType())
                        )
                        .build()

                    val request = Request.Builder()
                        .url(serverUrl.trimEnd('/') + "/analyze_file")
                        .post(requestBody)
                        .build()

                    httpClient.newCall(request).execute()
                }

                val body = response.body?.string()
                if (!response.isSuccessful || body == null) {
                    showError("Server error: ${response.code}")
                    return@launch
                }

                val json = JSONObject(body)
                showResult(json)

            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                showError("Error: ${e.message?.take(60)}")
            }
        }
    }

    private fun showResult(json: JSONObject) {
        val prediction = json.optString("prediction", "Unknown")
        val confidence = json.optDouble("confidence", 0.5)
        val frames = json.optInt("frames_analyzed", 0)
        val timeMs = json.optDouble("inference_time_ms", 0.0)
        val fileType = json.optString("file_type", "file")

        val isReal = prediction.equals("Real", true)
        val isFake = prediction.equals("Fake", true)

        val color = when {
            isReal -> Color.parseColor("#66BB6A")
            isFake -> Color.parseColor("#EF5350")
            else -> Color.parseColor("#FFA726")
        }
        val emoji = when { isReal -> "✅"; isFake -> "🚨"; else -> "⚠️" }

        txtVerdictEmoji.text = emoji
        txtVerdictLabel.text = prediction.uppercase()
        txtVerdictLabel.setTextColor(color)
        txtConfidence.text = "${"%.1f".format(confidence * 100)}% confident"
        txtResultDetails.text = "$frames ${fileType} frame${if (frames != 1) "s" else ""}  •  ${"%.0f".format(timeMs)}ms"

        txtAnalysisStatus.visibility = View.GONE
        resultCard.visibility = View.VISIBLE
        btnAnalyseFile.isEnabled = true

        // Reset verify state
        txtVerifyStatus.text = ""
        txtVerifyStatus.visibility = View.GONE
        btnMarkReal.isEnabled = true
        btnMarkFake.isEnabled = true
    }

    private fun showError(msg: String) {
        txtAnalysisStatus.text = "❌ $msg"
        btnAnalyseFile.isEnabled = true
    }

    // ─── User Verification (saves to server dataset) ───────────

    private fun verifyFile(label: String) {
        val uri = selectedFileUri ?: return
        val serverUrl = serverUrlInput.text.toString().trim()
        if (serverUrl.isBlank()) {
            Toast.makeText(this, "Enter server URL first", Toast.LENGTH_SHORT).show()
            return
        }

        scope.launch {
            btnMarkReal.isEnabled = false
            btnMarkFake.isEnabled = false
            txtVerifyStatus.text = "⏳ Saving to server…"
            txtVerifyStatus.visibility = View.VISIBLE

            try {
                val bytes = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use(InputStream::readBytes)
                } ?: throw Exception("Could not read file")

                val response = withContext(Dispatchers.IO) {
                    val requestBody = MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart(
                            "file", selectedFileName,
                            bytes.toRequestBody(selectedMimeType.toMediaType())
                        )
                        .addFormDataPart("label", label)
                        .build()

                    val request = Request.Builder()
                        .url(serverUrl.trimEnd('/') + "/verify")
                        .post(requestBody)
                        .build()

                    httpClient.newCall(request).execute()
                }

                val body = response.body?.string()
                if (!response.isSuccessful || body == null) {
                    txtVerifyStatus.text = "❌ Failed: ${response.code}"
                    btnMarkReal.isEnabled = true
                    btnMarkFake.isEnabled = true
                    return@launch
                }

                val json = JSONObject(body)
                val savedAs = json.optString("filename", selectedFileName)
                val color = if (label == "real") "#66BB6A" else "#EF5350"
                val icon = if (label == "real") "✅" else "🚨"

                txtVerifyStatus.text = "$icon Saved as $label: $savedAs"
                txtVerifyStatus.setTextColor(Color.parseColor(color))
                txtVerifyStatus.visibility = View.VISIBLE

                // Disable both buttons after successful verification
                btnMarkReal.isEnabled = false
                btnMarkFake.isEnabled = false
                btnMarkReal.alpha = 0.5f
                btnMarkFake.alpha = 0.5f

            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                txtVerifyStatus.text = "❌ ${e.message?.take(50)}"
                btnMarkReal.isEnabled = true
                btnMarkFake.isEnabled = true
            }
        }
    }

    // ─── Overlay Flow ──────────────────────────────────────────

    override fun onPause() {
        super.onPause()
        saveServerUrl()
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    private fun saveServerUrl() {
        val url = serverUrlInput.text.toString().trim()
        prefs.edit().putString(KEY_SERVER_URL, url).apply()
    }

    private fun startPermissionFlow() {
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.RECORD_AUDIO)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                needed.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
        if (needed.isNotEmpty()) {
            permissionLauncher.launch(needed.toTypedArray())
        } else {
            checkOverlayPermission()
        }
    }

    private fun checkOverlayPermission() {
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "Please enable 'Display over other apps'", Toast.LENGTH_LONG).show()
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST)
        } else {
            requestScreenCapture()
        }
    }

    @Suppress("DEPRECATION")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == OVERLAY_PERMISSION_REQUEST) {
            if (Settings.canDrawOverlays(this)) {
                requestScreenCapture()
            } else {
                Toast.makeText(this, "Overlay permission required", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun requestScreenCapture() {
        val manager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projectionLauncher.launch(manager.createScreenCaptureIntent())
    }

    private fun launchServiceWithOverlay(resultCode: Int, data: Intent) {
        val serviceIntent = Intent(this, CaptureService::class.java).apply {
            action = CaptureService.ACTION_SHOW_OVERLAY
            putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
            putExtra(CaptureService.EXTRA_RESULT_DATA, data)
        }
        startForegroundService(serviceIntent)
        Toast.makeText(this, "DeepGuard overlay active!", Toast.LENGTH_SHORT).show()
        moveTaskToBack(true)
    }
}
