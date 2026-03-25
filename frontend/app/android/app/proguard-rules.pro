# Proguard rules for Deepfake Capture app

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
-keep class okhttp3.** { *; }

# Keep data classes for JSON parsing
-keep class com.deepfake.capture.ApiClient$PredictionResult { *; }
