package com.codingwithmitch.essaimodel

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.codingwithmitch.essaimodel.ml.CnnLstmExp1Allclasses
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis
import org.apache.commons.math3.stat.descriptive.moment.Skewness
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation
import org.apache.commons.math3.stat.descriptive.rank.Median
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.pow
import kotlin.math.sqrt
import android.speech.tts.TextToSpeech
import java.util.Locale

class MainActivity : AppCompatActivity(), SensorEventListener, TextToSpeech.OnInitListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private lateinit var tts: TextToSpeech
    private var isTtsInitialized = false

    private val windowSize = 200
    private val accX = mutableListOf<Float>()
    private val accY = mutableListOf<Float>()
    private val accZ = mutableListOf<Float>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST)
        } ?: run {
            Log.e("MainActivity", "Accelerometer sensor not available")
        }

        // Initialize TextToSpeech
        tts = TextToSpeech(this, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts.setLanguage(Locale.ENGLISH)
            isTtsInitialized = result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED
            if (!isTtsInitialized) {
                Log.e("MainActivity", "TextToSpeech language is not supported")
            }
        } else {
            Log.e("MainActivity", "TextToSpeech initialization failed")
            isTtsInitialized = false
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            accX.add(event.values[0])
            accY.add(event.values[1])
            accZ.add(event.values[2])

            if (accX.size >= windowSize) {
                val window = Window(accX.toList(), accY.toList(), accZ.toList())

                // Normalize the window
                val meanX = 0.22124304f
                val meanY = 5.88213382f
                val meanZ = 0.70410258f
                val stdX = 3.96606625f
                val stdY = 7.09657627f
                val stdZ = 3.95054001f
                val normalizedWindow = normalizeWindow(window, meanX, meanY, meanZ, stdX, stdY, stdZ)

                // Calculate features
                val features = calculateFeatures(normalizedWindow)
                Log.d("MainActivity", "Calculated features: $features")

                // Calculate feature matrix
                val featureMatrix = calculateFeatureMatrix(normalizedWindow, features, windowSize)
                Log.d("MainActivity", "Feature Matrix: ${featureMatrix.joinToString { it.joinToString(", ") }}")

                // Verify the feature matrix shape
                if (featureMatrix.size == 200 && featureMatrix[0].size == 58) {
                    // Run model inference
                    try {
                        val modelOutput = runModelInference(featureMatrix)
                        Log.d("MainActivity", "Model Output: ${modelOutput.joinToString(", ")}")
                        handleModelOutput(modelOutput)
                    } catch (e: Exception) {
                        Log.e("MainActivity", "Model inference error: ${e.message}")
                    }
                } else {
                    Log.e("MainActivity", "Feature matrix has incorrect shape: ${featureMatrix.size}x${featureMatrix[0].size}")
                }

                // Clear lists for next window
                accX.clear()
                accY.clear()
                accZ.clear()
            }
        }
    }

    private fun handleModelOutput(modelOutput: FloatArray) {
        val classes = arrayOf("BSC", "CHU", "CSI", "CSO", "FKL", "FOL", "JOG", "JUM", "LYI", "SCH", "SDL", "SIT", "STD", "STN", "STU", "WAL")
        val maxIndex = modelOutput.indices.maxByOrNull { modelOutput[it] } ?: -1
        val result = if (maxIndex != -1) classes[maxIndex] else "unknown"

        Log.d("MainActivity", "Predicted activity: $result")

        if (result in arrayOf("FOL", "FKL", "BSC", "SDL")) {
            if (isTtsInitialized) {
                Log.d("MainActivity", "Speaking 'fall'")
                tts.speak("fall", TextToSpeech.QUEUE_FLUSH, null, null)
            } else {
                Log.e("MainActivity", "TextToSpeech not initialized")
            }
        } else {
            Log.d("MainActivity", "Predicted activity: $result")
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        // Not used in this example
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
        Log.d("MainActivity", "Activity destroyed and resources released")
    }

    data class Window(val accX: List<Float>, val accY: List<Float>, val accZ: List<Float>)

    private fun normalizeWindow(window: Window, meanX: Float, meanY: Float, meanZ: Float, stdX: Float, stdY: Float, stdZ: Float): Window {
        val normalizedAccX = window.accX.map { (it - meanX) / stdX }.map { if (it.isNaN()) 0f else it }
        val normalizedAccY = window.accY.map { (it - meanY) / stdY }.map { if (it.isNaN()) 0f else it }
        val normalizedAccZ = window.accZ.map { (it - meanZ) / stdZ }.map { if (it.isNaN()) 0f else it }
        return Window(normalizedAccX, normalizedAccY, normalizedAccZ)
    }

    private fun calculateFeatures(window: Window): Map<String, Float> {
        val features = mutableMapOf<String, Float>()

        val magnitude = window.accX.zip(window.accY).zip(window.accZ) { (x, y), z ->
            sqrt(x * x + y * y + z * z)
        }.map { if (it.isNaN()) 0f else it }

        val theta = window.accY.zip(window.accX) { y, x ->
            atan2(y, x)
        }.map { if (it.isNaN()) 0f else it }

        val axes = listOf("accX", "accY", "accZ")

        // Calculate per-axis features
        for (axis in axes) {
            val data = when (axis) {
                "accX" -> window.accX
                "accY" -> window.accY
                "accZ" -> window.accZ
                else -> throw IllegalArgumentException("Unknown axis: $axis")
            }
            val absData = data.map { abs(it) }.map { if (it.isNaN()) 0f else it }

            features["${axis}_mean"] = data.average().toFloat()
            features["${axis}_abs_mean"] = absData.average().toFloat()
            features["${axis}_median"] = safeEvaluate(Median(), data)
            features["${axis}_abs_median"] = safeEvaluate(Median(), absData)
            features["${axis}_std"] = safeEvaluate(StandardDeviation(), data)
            features["${axis}_abs_std"] = safeEvaluate(StandardDeviation(), absData)
            features["${axis}_skew"] = safeEvaluate(Skewness(), data)
            features["${axis}_abs_skew"] = safeEvaluate(Skewness(), absData)
            features["${axis}_kurtosis"] = safeEvaluate(Kurtosis(), data)
            features["${axis}_abs_kurtosis"] = safeEvaluate(Kurtosis(), absData)
            features["${axis}_min"] = data.minOrNull() ?: 0f
            features["${axis}_abs_min"] = absData.minOrNull() ?: 0f
            features["${axis}_max"] = data.maxOrNull() ?: 0f
            features["${axis}_abs_max"] = absData.maxOrNull() ?: 0f
        }

        // Features for magnitude and theta
        features["magnitude_mean"] = magnitude.average().toFloat()
        features["magnitude_std"] = safeEvaluate(StandardDeviation(), magnitude)
        features["theta_mean"] = theta.average().toFloat()
        features["theta_std"] = safeEvaluate(StandardDeviation(), theta)

        // Features only for theta
        features["theta_skew"] = safeEvaluate(Skewness(), theta)
        features["theta_kurtosis"] = safeEvaluate(Kurtosis(), theta)

        // Features only for magnitude
        features["magnitude_min"] = magnitude.minOrNull() ?: 0f
        features["magnitude_max"] = magnitude.maxOrNull() ?: 0f
        features["zero_crossing_rate"] = magnitude.zipWithNext().count { it.first * it.second < 0 }.toFloat() / magnitude.size
        features["diff_min_max"] = (magnitude.maxOrNull() ?: 0f) - (magnitude.minOrNull() ?: 0f)

        val time = window.accX.indices.map { it.toFloat() }
        val slope = calculateSlope(time, magnitude)
        features["slope"] = if (slope.isNaN()) 0f else slope
        features["abs_slope"] = abs(features["slope"] ?: 0f)
        features["avg_acc_rate"] = magnitude.zipWithNext { a, b -> abs(b - a) }.average().toFloat()

        return features
    }

    private fun safeEvaluate(stat: Any, data: List<Float>): Float {
        return try {
            when (stat) {
                is Median -> stat.evaluate(data.map { it.toDouble() }.toDoubleArray()).toFloat()
                is StandardDeviation -> if (data.size > 1) stat.evaluate(data.map { it.toDouble() }.toDoubleArray()).toFloat() else 0f
                is Skewness -> if (data.size > 1) stat.evaluate(data.map { it.toDouble() }.toDoubleArray()).toFloat() else 0f
                is Kurtosis -> if (data.size > 1) stat.evaluate(data.map { it.toDouble() }.toDoubleArray()).toFloat() else 0f
                else -> 0f
            }
        } catch (e: Exception) {
            0f
        }
    }

    private fun calculateSlope(time: List<Float>, axis: List<Float>): Float {
        val validTime = time.map { if (it.isNaN()) 0f else it }
        val validAxis = axis.map { if (it.isNaN()) 0f else it }

        val meanX = validTime.average().toFloat()
        val meanY = validAxis.average().toFloat()

        val numerator = validTime.zip(validAxis).sumOf { (x, y) -> (x - meanX) * (y - meanY).toDouble() }
        val denominator = validTime.sumOf { x -> ((x - meanX).pow(2)).toDouble() }

        return if (denominator != 0.0) (numerator / denominator).toFloat() else 0f
    }

    private fun calculateFeatureMatrix(window: Window, features: Map<String, Float>, windowSize: Int = 200): Array<FloatArray> {
        val featureMatrix = Array(windowSize) { FloatArray(features.size + 3) }

        // Fill first three columns with absolute values of the data points
        for (i in 0 until windowSize) {
            featureMatrix[i][0] = abs(window.accX[i])
            featureMatrix[i][1] = abs(window.accY[i])
            featureMatrix[i][2] = abs(window.accZ[i])
        }

        // Repeat the features for each row
        features.values.forEachIndexed { index, value ->
            for (i in 0 until windowSize) {
                featureMatrix[i][index + 3] = value
            }
        }

        return featureMatrix
    }

    private fun runModelInference(featureMatrix: Array<FloatArray>): FloatArray {
        try {
            val model = CnnLstmExp1Allclasses.newInstance(this)

            // Prepare the input buffer
            val byteBuffer = ByteBuffer.allocateDirect(4 * windowSize * featureMatrix[0].size)
            byteBuffer.order(ByteOrder.nativeOrder())
            for (i in featureMatrix.indices) {
                for (j in featureMatrix[i].indices) {
                    byteBuffer.putFloat(featureMatrix[i][j])
                }
            }

            // Creates inputs for reference
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, windowSize, featureMatrix[0].size), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            // Releases model resources if no longer used
            model.close()

            return outputFeature0.floatArray
        } catch (e: Exception) {
            Log.e("MainActivity", "Model inference error: ${e.message}")
            throw e
        }
    }
}
