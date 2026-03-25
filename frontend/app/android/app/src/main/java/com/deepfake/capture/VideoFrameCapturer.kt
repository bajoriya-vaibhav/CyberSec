package com.deepfake.capture

import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import java.io.ByteArrayOutputStream

/**
 * Captures screen frames from a VirtualDisplay backed by MediaProjection.
 * Samples frames and resizes to 224x224 JPEG.
 */
class VideoFrameCapturer(
    private val mediaProjection: MediaProjection,
    private val screenWidth: Int,
    private val screenHeight: Int,
    private val screenDensity: Int
) {
    companion object {
        private const val TAG = "VideoFrameCapturer"
        private const val OUTPUT_SIZE = 720
        private const val JPEG_QUALITY = 95
    }

    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null

    @Volatile
    private var latestFrame: ByteArray? = null

    fun start() {
        Log.i(TAG, "Starting with ${screenWidth}x${screenHeight} density=$screenDensity")

        handlerThread = HandlerThread("FrameCaptureThread").also { it.start() }
        handler = Handler(handlerThread!!.looper)

        imageReader = ImageReader.newInstance(
            screenWidth, screenHeight, PixelFormat.RGBA_8888, 2
        )
        Log.i(TAG, "ImageReader created")

        imageReader!!.setOnImageAvailableListener({ reader ->
            var image: Image? = null
            try {
                image = reader.acquireLatestImage()
                if (image == null) return@setOnImageAvailableListener

                val bitmap = imageToBitmap(image)
                if (bitmap == null) {
                    Log.w(TAG, "imageToBitmap returned null")
                    return@setOnImageAvailableListener
                }

                val resized = Bitmap.createScaledBitmap(bitmap, OUTPUT_SIZE, OUTPUT_SIZE, true)
                if (resized != bitmap) bitmap.recycle()

                val outputStream = ByteArrayOutputStream()
                resized.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, outputStream)
                resized.recycle()

                latestFrame = outputStream.toByteArray()
                Log.d(TAG, "Frame captured: ${latestFrame!!.size} bytes")
            } catch (e: Exception) {
                Log.e(TAG, "Error capturing frame", e)
            } finally {
                try {
                    image?.close()
                } catch (e: Exception) {
                    // ignore
                }
            }
        }, handler)

        virtualDisplay = mediaProjection.createVirtualDisplay(
            "DeepfakeCapture",
            screenWidth, screenHeight, screenDensity,
            0,
            imageReader!!.surface,
            null, null
        )

        Log.i(TAG, "VirtualDisplay created, capture running")
    }

    fun consumeLatestFrame(): ByteArray? {
        val frame = latestFrame
        latestFrame = null
        return frame
    }

    fun stop() {
        try { virtualDisplay?.release() } catch (e: Exception) { Log.w(TAG, "Error releasing VirtualDisplay", e) }
        virtualDisplay = null
        try { imageReader?.close() } catch (e: Exception) { Log.w(TAG, "Error closing ImageReader", e) }
        imageReader = null
        handlerThread?.quitSafely()
        handlerThread = null
        handler = null
        latestFrame = null
        Log.i(TAG, "Video frame capture stopped")
    }

    private fun imageToBitmap(image: Image): Bitmap? {
        return try {
            val plane = image.planes[0]
            val buffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * screenWidth

            val bitmapWidth = screenWidth + rowPadding / pixelStride
            val bitmap = Bitmap.createBitmap(
                bitmapWidth, screenHeight, Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)

            if (rowPadding > 0) {
                val cropped = Bitmap.createBitmap(bitmap, 0, 0, screenWidth, screenHeight)
                bitmap.recycle()
                cropped
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "imageToBitmap error", e)
            null
        }
    }
}
