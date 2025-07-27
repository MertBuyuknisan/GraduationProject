package com.mertbuyuknisan.bitirmeprojesi

import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.github.mikephil.charting.components.AxisBase
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.github.mikephil.charting.formatter.ValueFormatter
import com.mertbuyuknisan.bitirmeprojesi.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var predictionResponse: PredictionResponse? = null
    private val modelNames = listOf("Lineer Regresyon", "KNN", "Random Forest", "LSTM")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupSpinner()
        fetchPredictionData()

        binding.buttonShowGraph.setOnClickListener {
            val selectedModel = binding.spinnerModels.selectedItem.toString()
            updateChart(selectedModel)
        }
    }

    private fun setupSpinner() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, modelNames)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModels.adapter = adapter
    }

    private fun fetchPredictionData() {
        binding.progressBar.visibility = View.VISIBLE
        binding.lineChart.visibility = View.INVISIBLE

        lifecycleScope.launch {
            try {
                val response = RetrofitClient.instance.getPredictions()
                binding.progressBar.visibility = View.GONE
                binding.lineChart.visibility = View.VISIBLE

                if (response.isSuccessful) {
                    predictionResponse = response.body()
                    if (modelNames.isNotEmpty()) {
                        updateChart(modelNames.first())
                    }
                } else {
                    showError("Veri alınamadı.Hata Kodu: ${response.code()}")
                }
            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                showError("Ağ hatası: ${e.message}.")
            }
        }
    }

    private fun updateChart(selectedModel: String) {
        val response = predictionResponse ?: run {
            showError("Grafik verisi henüz yüklenmedi.")
            return
        }

        val actualEntries = ArrayList<Entry>()
        response.actualPrices.forEachIndexed { index, price ->
            actualEntries.add(Entry(index.toFloat(), price))
        }
        val actualDataSet = LineDataSet(actualEntries, "Gerçek Fiyat").apply {
            color = Color.rgb(255, 165, 0) // Orange
            setDrawValues(false)
            setDrawCircles(false)
            lineWidth = 2f
        }

        val predictedPrices = when (selectedModel) {
            "Lineer Regresyon" -> response.predictions.linearRegression
            "KNN" -> response.predictions.knn
            "Random Forest" -> response.predictions.randomForest
            "LSTM" -> response.predictions.lstm
            else -> emptyList()
        }

        val predictedEntries = ArrayList<Entry>()
        predictedPrices.forEachIndexed { index, price ->
            predictedEntries.add(Entry(index.toFloat(), price))
        }
        val predictedDataSet = LineDataSet(predictedEntries, "$selectedModel Tahmini").apply {
            color = getModelColor(selectedModel)
            setDrawValues(false)
            setDrawCircles(false)
            lineWidth = 2f
            enableDashedLine(10f, 5f, 0f)
        }

        binding.lineChart.data = LineData(actualDataSet, predictedDataSet)
        setupChartStyle()
        binding.lineChart.invalidate()
    }

    private fun setupChartStyle() {
        binding.lineChart.apply {
            description.text = "BTC/USD Fiyat Tahmini"
            description.textSize = 12f
            setDrawGridBackground(false)
            xAxis.valueFormatter = MyXAxisValueFormatter(predictionResponse?.testDates ?: emptyList())
            xAxis.position = XAxis.XAxisPosition.BOTTOM
            xAxis.granularity = 1f
            xAxis.setLabelCount(4, true)
            axisRight.isEnabled = false
            legend.isEnabled = true
        }
    }

    private fun getModelColor(modelName: String): Int {
        return when (modelName) {
            "Lineer Regresyon" -> Color.rgb(34, 139, 34)
            "KNN" -> Color.rgb(128, 0, 128)
            "Random Forest" -> Color.rgb(220, 20, 60)
            "LSTM" -> Color.rgb(0, 0, 255)
            else -> Color.GRAY
        }
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }
}

class MyXAxisValueFormatter(private val dates: List<String>) : ValueFormatter() {
    private val inputFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
    private val outputFormat = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())

    override fun getAxisLabel(value: Float, axis: AxisBase?): String {
        val index = value.toInt()
        return if (index >= 0 && index < dates.size) {
            try {
                val date = inputFormat.parse(dates[index])
                date?.let { outputFormat.format(it) } ?: index.toString()
            } catch (e: Exception) {
                index.toString()
            }
        } else {
            ""
        }
    }
}