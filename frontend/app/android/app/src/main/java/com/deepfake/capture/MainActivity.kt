package com.deepfake.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * Main settings screen for DeepGuard.
 *
 * Configures:
 *   - Server URL (saved to SharedPrefs, read by overlay/service)
 *   - Analysis duration (3-30 seconds per cycle)
 *
 * On "Enable Overlay": requests permissions → starts CaptureService → minimizes.
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
    private lateinit var prefs: SharedPreferences

    private lateinit var serverUrlInput: EditText
    private lateinit var durationSeekBar: SeekBar
    private lateinit var durationValueText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)

        // Bind UI
        serverUrlInput = findViewById(R.id.editServerUrl)
        durationSeekBar = findViewById(R.id.seekDuration)
        durationValueText = findViewById(R.id.txtDurationValue)

        // Restore saved settings
        val savedUrl = prefs.getString(KEY_SERVER_URL, "http://10.0.2.2:7860") ?: ""
        serverUrlInput.setText(savedUrl)

        val savedDuration = prefs.getInt(KEY_ANALYSIS_DURATION, 15)
        durationSeekBar.progress = savedDuration
        durationValueText.text = "$savedDuration seconds"

        // Duration slider listener
        durationSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress.coerceAtLeast(3) // Minimum 3 seconds
                durationValueText.text = "$value seconds"
                prefs.edit().putInt(KEY_ANALYSIS_DURATION, value).apply()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Auto-save URL on focus change
        serverUrlInput.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) saveServerUrl()
        }

        // Register launchers
        projectionLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                saveServerUrl() // Save before launching service
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

        // Enable Overlay button
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
    }

    override fun onPause() {
        super.onPause()
        saveServerUrl()
    }

    private fun saveServerUrl() {
        val url = serverUrlInput.text.toString().trim()
        prefs.edit().putString(KEY_SERVER_URL, url).apply()
    }

    // ─── Permission Flow ───────────────────────────────────────

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
