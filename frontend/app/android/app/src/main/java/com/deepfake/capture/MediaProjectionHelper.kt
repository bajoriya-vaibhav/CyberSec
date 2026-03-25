package com.deepfake.capture

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.util.Log

/**
 * Wraps MediaProjection setup, permission flow, and lifecycle management.
 */
class MediaProjectionHelper(private val context: Context) {

    companion object {
        private const val TAG = "MediaProjectionHelper"
    }

    private var mediaProjection: MediaProjection? = null

    /**
     * Returns the MediaProjectionManager to create a screen capture intent.
     */
    fun getProjectionManager(): MediaProjectionManager {
        return context.getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    }

    /**
     * Creates a MediaProjection from the activity result data.
     * Call this after the user grants screen recording permission.
     */
    fun createProjection(resultCode: Int, data: Intent): MediaProjection? {
        val manager = getProjectionManager()
        mediaProjection = manager.getMediaProjection(resultCode, data)

        mediaProjection?.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                Log.i(TAG, "MediaProjection stopped by system")
            }
        }, null)

        Log.i(TAG, "MediaProjection created successfully")
        return mediaProjection
    }

    fun getProjection(): MediaProjection? = mediaProjection

    fun stop() {
        mediaProjection?.stop()
        mediaProjection = null
        Log.i(TAG, "MediaProjection released")
    }
}
