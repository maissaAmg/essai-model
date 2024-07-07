package com.codingwithmitch.essaimodel

import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.EditText
import androidx.appcompat.app.AppCompatActivity
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import java.util.*

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var adlEditText: EditText
    private lateinit var fallEditText: EditText

    private lateinit var textToSpeech: TextToSpeech
    private var serviceBound = false
    private lateinit var serviceConnection: ServiceConnection
    private lateinit var resultReceiver: BroadcastReceiver

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textToSpeech = TextToSpeech(this, this).apply {
            language = Locale.US
        }

        adlEditText = findViewById(R.id.ADL)
        fallEditText = findViewById(R.id.FALL)

        // Service connection setup
        serviceConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                serviceBound = true
                Log.d("MainActivity", "Service connected")
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                serviceBound = false
                Log.d("MainActivity", "Service disconnected")
            }
        }

        // Bind to the inference service
        InferenceService.bindService(this, serviceConnection)

        // Initialize and register the BroadcastReceiver to receive inference results
        resultReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val result = intent.getFloatArrayExtra("result")
                result?.let {
                    Log.d("MainActivity", "Inference result received: ${it.contentToString()}")
                    updateUIWithResults(it)
                } ?: Log.d("MainActivity", "No result received")
            }
        }

        LocalBroadcastManager.getInstance(this)
            .registerReceiver(resultReceiver, IntentFilter("com.codingwithmitch.essaimodel.INFERENCE_RESULT"))
    }

    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) {
            Log.e("MainActivity", "TextToSpeech initialization failed")
        } else {
            Log.d("MainActivity", "TextToSpeech initialized successfully")
        }
    }

    // Update UI with inference results
    private fun updateUIWithResults(result: FloatArray) {
        if (result.size >= 2) {
            val fallValue = result[0]
            val adlValue = result[1]

            Log.d("MainActivity", "Updating UI with results: fallValue=$fallValue, adlValue=$adlValue")
            adlEditText.setText(adlValue.toString())
            fallEditText.setText(fallValue.toString())

            if (fallValue > adlValue) {
                speak("Fall detected")
            }
        } else {
            Log.e("MainActivity", "Invalid result size: ${result.size}")
        }
    }

    private fun speak(text: String) {
        Log.d("MainActivity", "Speaking text: $text")
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (serviceBound) {
            InferenceService.unbindService(this, serviceConnection)
            serviceBound = false
        }
        textToSpeech.stop()
        textToSpeech.shutdown()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(resultReceiver)
        Log.d("MainActivity", "Activity destroyed and resources released")
    }
}
