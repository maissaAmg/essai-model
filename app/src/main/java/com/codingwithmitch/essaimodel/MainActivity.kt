package com.codingwithmitch.essaimodel

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
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


    // Déclaration d'un BroadcastReceiver pour recevoir les résultats de l'inférence.
    private lateinit var resultReceiver: BroadcastReceiver

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textToSpeech = TextToSpeech(this, this)
        textToSpeech.language = Locale.US


        adlEditText = findViewById(R.id.ADL)
        fallEditText = findViewById(R.id.FALL)


        // Lancement du service d'inférence
        val serviceIntent = Intent(this, InferenceService::class.java)
        startService(serviceIntent)


        // Initialisation et enregistrement du BroadcastReceiver pour recevoir les résultats.
        resultReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                // Obtention des résultats de l'inférence et mise à jour de l'interface utilisateur.
                val result = intent.getFloatArrayExtra("result")
                result?.let { updateUIWithResults(it) }
            }
        }

        // Enregistrement du BroadcastReceiver pour écouter les résultats du service d'inférence.
        LocalBroadcastManager.getInstance(this)
            .registerReceiver(resultReceiver, IntentFilter("INFERENCESERVICE_RESULT"))


    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            // TextToSpeech initialization successful
        } else {
            // TextToSpeech initialization failed
        }
    }


    // Méthode pour mettre à jour l'interface utilisateur avec les résultats de l'inférence.
    private fun updateUIWithResults(result: FloatArray) {
        if (result.size >= 2) {

            val adlValue = result[0]
            val fallValue = result[1]

            adlEditText.setText(adlValue.toString())
            fallEditText.setText(fallValue.toString())


            // Check if fall value is greater than ADL value
            if (fallValue > adlValue) {
                // Trigger voice prompt here
                Log.d("*******","************************")
                Log.d("VoiceTrigger", "A fall has occured")
                Log.d("*******","************************")
                speak("Fall detected")
            }
        }
    }

    private fun speak(text: String) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_ADD, null, null)
    }


    // Méthode onDestroy appelée lors de la destruction de l'activité.
    override fun onDestroy() {
        super.onDestroy()
        textToSpeech.stop()
        textToSpeech.shutdown()
        // Unregister the BroadcastReceiver
        LocalBroadcastManager.getInstance(this).unregisterReceiver(resultReceiver)
    }

}
