package com.codingwithmitch.essaimodel

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.EditText
import androidx.localbroadcastmanager.content.LocalBroadcastManager


class MainActivity : AppCompatActivity() {

    private lateinit var adlEditText: EditText
    private lateinit var fallEditText: EditText

    // Déclaration d'un BroadcastReceiver pour recevoir les résultats de l'inférence.
    private lateinit var resultReceiver: BroadcastReceiver

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


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

    // Méthode pour mettre à jour l'interface utilisateur avec les résultats de l'inférence.
    private fun updateUIWithResults(result: FloatArray) {
        if (result.size >= 2) {
            adlEditText.setText(result[0].toString())
            fallEditText.setText(result[1].toString())
        }
    }

    // Méthode onDestroy appelée lors de la destruction de l'activité.
    override fun onDestroy() {
        super.onDestroy()
        // Unregister the BroadcastReceiver
        LocalBroadcastManager.getInstance(this).unregisterReceiver(resultReceiver)
    }

}
