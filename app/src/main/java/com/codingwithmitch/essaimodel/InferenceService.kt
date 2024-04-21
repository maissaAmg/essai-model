package com.codingwithmitch.essaimodel

import android.app.Service
import android.app.Service.START_STICKY
import android.content.Context
import android.content.Intent
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


// Classe principale pour le service d'inférence qui utilise les données du capteur.
class InferenceService : Service(), SensorEventListener {

    // Déclarations et initialisations des variables.
    private lateinit var interpreter: Interpreter  // L'interpréteur pour le modèle TensorFlow Lite.
    private lateinit var sensorManager: SensorManager  // Gère les capteurs du dispositif.
    private var accelerometerSensor: Sensor? = null  // Capteur d'accéléromètre.
    private val accelerometerData = mutableListOf<FloatArray>()  // Stocke les données de l'accéléromètre.
    private var collectionStartTime: Long = 0  // Marque le début de la collecte des données.
    // Constantes pour la normalisation des données.
    private val MEANS = floatArrayOf(-0.00587745f, -0.60819228f, -0.09736887f)
    private val STANDARD_DEVIATIONS = floatArrayOf(0.40432001f, 0.66772087f, 0.4787974f)

    // Liaison du service, généralement non utilisée dans ce contexte.
    override fun onBind(intent: Intent): IBinder? {
        return null
    }


    // Initialisation lors de la création du service.
    override fun onCreate() {
        super.onCreate()
        initTFLiteInterpreter() // Initialisation de l'interpréteur TensorFlow Lite.
        // Configuration et enregistrement du capteur d'accéléromètre.
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        sensorManager.registerListener(this, accelerometerSensor, SensorManager.SENSOR_DELAY_FASTEST)
    }

    // Démarrage du service.
    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {
        // Le service est démarré et s'exécutera en arrière-plan.
        return START_STICKY
    }

    // Gestion des changements de données du capteur.
    override fun onSensorChanged(event: SensorEvent) {
        // Traite uniquement les données de l'accéléromètre
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            // Logique de collecte et traitement des données.
            val currentTime = System.currentTimeMillis()

            if (accelerometerData.isEmpty()) {
                collectionStartTime = currentTime
            }

            if (currentTime - collectionStartTime < 1000) {
                val standardizedValues = standardizeValues(event.values)
                accelerometerData.add(standardizedValues)
            } else {
                // Exécution de l'inférence du modèle et gestion du cycle de vie des données.
                Log.d("SensorDataCollection", "1 second passed, processing data")
                val sampledData = sampleData(accelerometerData)
                val byteBufferSensor = createByteBufferFromSampledData(sampledData)
                runModelInference(byteBufferSensor)

                accelerometerData.clear()
                collectionStartTime = currentTime
                Log.d("SensorDataCollection", "Data buffer cleared for next collection")
            }
        }
    }

    // Méthode pour exécuter l'inférence du modèle.
    private fun runModelInference(byteBuffer: ByteBuffer) {
        // Préparation et exécution de l'inférence.
        Log.d("ModelInference", "Running model inference")
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 50, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        val outputs = mutableMapOf<Int, Any>()
        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 2), DataType.FLOAT32)
        outputs[0] = outputFeature0.buffer

        interpreter.runForMultipleInputsOutputs(arrayOf(inputFeature0.buffer), outputs)
        val result = outputFeature0.floatArray
        sendInferenceResults(result) //envoyer les données
        Log.d("ModelInference", "Model inference complete. Result: ${result.contentToString()}")

    }

    // Échantillonnage des données collectées.
    private fun sampleData(data: MutableList<FloatArray>): List<FloatArray> {
        val interval = data.size / 50
        return data.filterIndexed { index, _ -> index % interval == 0 }.take(50)
    }

    // Normalisation des valeurs des données.
    private fun standardizeValues(values: FloatArray): FloatArray {
        return floatArrayOf(
            (values[0] - MEANS[0]) / STANDARD_DEVIATIONS[0],
            (values[1] - MEANS[1]) / STANDARD_DEVIATIONS[1],
            (values[2] - MEANS[2]) / STANDARD_DEVIATIONS[2]
        )
    }

    // Création d'un ByteBuffer à partir des données échantillonnées.
    private fun createByteBufferFromSampledData(sampledData: List<FloatArray>): ByteBuffer {
        val numSamples = 50
        val numFeatures = 3
        val bytesPerFloat = 4

        val byteBuffer = ByteBuffer.allocateDirect(numSamples * numFeatures * bytesPerFloat)
        byteBuffer.order(ByteOrder.nativeOrder())

        sampledData.forEach { dataPoint ->
            byteBuffer.putFloat(dataPoint[0])
            byteBuffer.putFloat(dataPoint[1])
            byteBuffer.putFloat(dataPoint[2])
        }

        Log.d("ByteBuffer", "ByteBuffer created from sampled data")
        return byteBuffer
    }

    // Initialisation de l'interpréteur TensorFlow Lite.
    private fun initTFLiteInterpreter() {
        val modelByteBuffer = loadModelFile("lstm0_model_50.tflite")
        val options = Interpreter.Options().addDelegate(FlexDelegate())
        interpreter = Interpreter(modelByteBuffer, options)
        Log.d("TFLite", "TensorFlow Lite Interpreter initialized")
    }

    // Chargement du fichier modèle TensorFlow Lite.
    private fun loadModelFile(modelName: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength).also {
            Log.d("LoadModel", "Model file loaded successfully")
        }
    }

    // Gestion des changements de précision du capteur (peut rester vide si non nécessaire).
    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
    }

    // Envoi des résultats de l'inférence.
    private fun sendInferenceResults(result: FloatArray) {
        Intent("INFERENCESERVICE_RESULT").apply {
            putExtra("result", result)
            LocalBroadcastManager.getInstance(this@InferenceService).sendBroadcast(this)
        }
    }

    // Nettoyage lors de la destruction du service.
    override fun onDestroy() {
        sensorManager.unregisterListener(this)
        interpreter.close()
        super.onDestroy()
        Log.d("MainActivity", "Resources released and Activity destroyed")
    }

}