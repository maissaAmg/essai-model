package com.codingwithmitch.essaimodel

import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import android.util.Log
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.*

// Service pour gérer l'inférence du modèle TensorFlow Lite avec les données des capteurs.
class InferenceService : Service(), SensorEventListener {

    // Déclarations de variables
    private lateinit var interpreter: Interpreter // Interpréteur TensorFlow Lite.
    private lateinit var sensorManager: SensorManager // Gestionnaire de capteurs.
    private var accelerometerSensor: Sensor? = null // Capteur d'accéléromètre.
    private val accelerometerData = mutableListOf<FloatArray>() // Liste des données de l'accéléromètre.

    // Méthode appelée lors de la liaison du service (non utilisée ici).
    override fun onBind(intent: Intent): IBinder? {
        return null
    }

    // Méthode appelée lors de la création du service.
    override fun onCreate() {
        super.onCreate()
        Log.d("InferenceService", "Service created")
        initTFLiteInterpreter() // Initialiser l'interpréteur TensorFlow Lite.
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        sensorManager.registerListener(this, accelerometerSensor, SensorManager.SENSOR_DELAY_FASTEST)
    }

    // Méthode appelée lors du démarrage du service.
    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {
        Log.d("InferenceService", "Service started")
        return START_STICKY
    }

    // Méthode appelée lorsque les données du capteur changent.
    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            Log.d("SensorEvent", "Accelerometer data received: ${event.values.contentToString()}")
            accelerometerData.add(event.values.copyOf())
            if (accelerometerData.size >= 200) {
                Log.d("SensorEvent", "Collected 200 data points")
                val sampledData = sampleData(accelerometerData)
                val byteBufferSensor = createByteBufferFromSampledData(sampledData)
                runModelInference(byteBufferSensor)
                accelerometerData.clear()
                Log.d("SensorEvent", "Accelerometer data cleared")
            }
        }
    }

    // Méthode pour exécuter l'inférence du modèle.
    private fun runModelInference(byteBuffer: ByteBuffer) {
        Log.d("ModelInference", "Running model inference")
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 200, 58), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 2), DataType.FLOAT32)
        interpreter.run(inputFeature0.buffer, outputFeature0.buffer)

        val result = outputFeature0.floatArray
        Log.d("ModelInference", "Inference result: ${result.contentToString()}")
        sendInferenceResults(result)
    }

    // Méthode pour échantillonner les données collectées.
    private fun sampleData(data: List<FloatArray>): List<FloatArray> {
        Log.d("SampleData", "Sampling data")
        return data.take(200)
    }

    // Méthode pour normaliser les données.
    private fun normalizeData(data: FloatArray): FloatArray {
        Log.d("NormalizeData", "Normalizing data: ${data.contentToString()}")
        val means = floatArrayOf(-0.2455672f, 7.21737844f, 1.3798924f)
        val stdDevs = floatArrayOf(4.06080787f, 5.21834111f, 4.21955204f)
        return floatArrayOf(
            (data[0] - means[0]) / stdDevs[0],
            (data[1] - means[1]) / stdDevs[1],
            (data[2] - means[2]) / stdDevs[2]
        ).also { Log.d("NormalizeData", "Normalized data: ${it.contentToString()}") }
    }

    // Méthode pour calculer les caractéristiques des données.
    private fun calculateFeatures(window: List<FloatArray>): FloatArray {
        Log.d("CalculateFeatures", "Calculating features for data window")
        val features = FloatArray(58)
        val magnitude = window.map { sqrt(it[0] * it[0] + it[1] * it[1] + it[2] * it[2]) }
        val theta = window.map { atan2(it[1], it[0]) }

        for ((i, sample) in window.withIndex()) {
            val absValues = sample.map { abs(it) }.toFloatArray()
            if (i * 3 + 3 <= features.size) {
                System.arraycopy(absValues, 0, features, i * 3, 3)
            }
        }

        val xValues = window.map { it[0] }
        val yValues = window.map { it[1] }
        val zValues = window.map { it[2] }
        val axes = listOf(xValues, yValues, zValues)

        for ((axisIndex, axis) in axes.withIndex()) {
            val baseIndex = 9 + axisIndex * 14
            features[baseIndex] = axis.mean()
            features[baseIndex + 1] = axis.absMean()
            features[baseIndex + 2] = axis.median()
            features[baseIndex + 3] = axis.absMedian()
            features[baseIndex + 4] = axis.std()
            features[baseIndex + 5] = axis.map { abs(it) }.std()
            features[baseIndex + 6] = axis.skew()
            features[baseIndex + 7] = axis.map { abs(it) }.skew()
            features[baseIndex + 8] = axis.kurtosis()
            features[baseIndex + 9] = axis.map { abs(it) }.kurtosis()
            features[baseIndex + 10] = axis.minOrNull() ?: 0f
            features[baseIndex + 11] = axis.map { abs(it) }.minOrNull() ?: 0f
            features[baseIndex + 12] = axis.maxOrNull() ?: 0f
            features[baseIndex + 13] = axis.map { abs(it) }.maxOrNull() ?: 0f
        }

        features[51] = magnitude.mean()
        features[52] = magnitude.std()
        features[53] = theta.mean()
        features[54] = theta.std()
        features[55] = theta.skew()
        features[56] = theta.kurtosis()
        features[57] = magnitude.maxOrNull()?.minus(magnitude.minOrNull() ?: 0f) ?: 0f

        Log.d("CalculateFeatures", "Calculated features: ${features.contentToString()}")
        return features
    }

    // Méthode pour créer un ByteBuffer à partir des données échantillonnées.
    private fun createByteBufferFromSampledData(sampledData: List<FloatArray>): ByteBuffer {
        Log.d("CreateByteBuffer", "Creating ByteBuffer from sampled data")
        val windowFeatures = sampledData.map { normalizeData(it) }
        val features = calculateFeatures(windowFeatures)

        val byteBuffer = ByteBuffer.allocateDirect(200 * 58 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        features.forEach { byteBuffer.putFloat(it) }
        Log.d("CreateByteBuffer", "ByteBuffer created: ${byteBufferToString(byteBuffer)}")
        return byteBuffer
    }

    // Méthode pour convertir un ByteBuffer en chaîne de caractères (pour les logs).
    private fun byteBufferToString(buffer: ByteBuffer): String {
        buffer.rewind()
        val floats = FloatArray(buffer.remaining() / 4)
        buffer.asFloatBuffer().get(floats)
        return floats.joinToString(prefix = "[", postfix = "]", separator = ", ")
    }

    // Méthode pour initialiser l'interpréteur TensorFlow Lite.
    private fun initTFLiteInterpreter() {
        try {
            val modelByteBuffer = loadModelFile("cnn_lstm_exp2.tflite")
            val options = Interpreter.Options().addDelegate(FlexDelegate())
            interpreter = Interpreter(modelByteBuffer, options)
            Log.d("TFLite", "TensorFlow Lite Interpreter initialized successfully")
        } catch (e: Exception) {
            Log.e("TFLite", "Error initializing TensorFlow Lite Interpreter", e)
        }
    }

    // Méthode pour charger le fichier du modèle.
    private fun loadModelFile(modelName: String): ByteBuffer {
        return try {
            val fileDescriptor = assets.openFd(modelName)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            Log.d("LoadModel", "Model file start offset: $startOffset, declared length: $declaredLength")
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).also {
                Log.d("LoadModel", "Model file loaded successfully: $modelName")
            }
        } catch (e: Exception) {
            Log.e("LoadModel", "Error loading model file: $modelName", e)
            throw e
        }
    }

    // Méthode appelée lorsque la précision du capteur change (non utilisée ici).
    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}

    // Méthode pour envoyer les résultats de l'inférence.
    private fun sendInferenceResults(result: FloatArray) {
        Intent("com.codingwithmitch.essaimodel.INFERENCE_RESULT").apply {
            putExtra("result", result)
            LocalBroadcastManager.getInstance(this@InferenceService).sendBroadcast(this)
            Log.d("SendInferenceResults", "Inference results sent: ${result.contentToString()}")
        }
    }

    // Méthode appelée lorsque le service est détruit.
    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        interpreter.close()
        super.onDestroy()
        Log.d("InferenceService", "Service destroyed")
    }

    companion object {
        // Méthode pour lier le service.
        fun bindService(context: Context, connection: ServiceConnection) {
            val intent = Intent(context, InferenceService::class.java)
            context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
            Log.d("BindService", "Service bound")
        }

        // Méthode pour délier le service.
        fun unbindService(context: Context, connection: ServiceConnection) {
            context.unbindService(connection)
            Log.d("UnbindService", "Service unbound")
        }
    }
}

// Fonctions d'extension pour List<Float>
fun List<Float>.mean() = this.sum() / this.size
fun List<Float>.std() = sqrt(this.map { (it - this.mean()).pow(2) }.sum() / this.size)
fun List<Float>.absMean() = this.map { abs(it) }.mean()
fun List<Float>.median(): Float {
    val sorted = this.sorted()
    return if (sorted.size % 2 == 0) {
        (sorted[sorted.size / 2] + sorted[sorted.size / 2 - 1]) / 2
    } else {
        sorted[sorted.size / 2]
    }
}
fun List<Float>.absMedian() = this.map { abs(it) }.median()
fun List<Float>.skew(): Float {
    val mean = this.mean()
    val n = this.size
    val m3 = this.sumByDouble { ((it - mean).pow(3)).toDouble() } / n
    val m2 = this.sumByDouble { ((it - mean).pow(2)).toDouble() } / n
    return (m3 / m2.pow(1.5)).toFloat()
}
fun List<Float>.kurtosis(): Float {
    val mean = this.mean()
    val n = this.size
    val m4 = this.sumByDouble { ((it - mean).pow(4)).toDouble() } / n
    val m2 = this.sumByDouble { ((it - mean).pow(2)).toDouble() } / n
    return (m4 / m2.pow(2)).toFloat()
}
