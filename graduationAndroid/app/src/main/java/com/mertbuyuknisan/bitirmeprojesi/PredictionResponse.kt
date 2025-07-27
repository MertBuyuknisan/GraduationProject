package com.mertbuyuknisan.bitirmeprojesi

import com.google.gson.annotations.SerializedName

data class PredictionResponse(
    @SerializedName("test_dates")
    val testDates: List<String>,

    @SerializedName("actual_prices")
    val actualPrices: List<Float>,

    @SerializedName("predictions")
    val predictions: Predictions
)

data class Predictions(
    @SerializedName("Lineer Regresyon")
    val linearRegression: List<Float>,

    @SerializedName("KNN")
    val knn: List<Float>,

    @SerializedName("Random Forest")
    val randomForest: List<Float>,

    @SerializedName("LSTM")
    val lstm: List<Float>
)

