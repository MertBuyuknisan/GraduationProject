package com.mertbuyuknisan.bitirmeprojesi

import retrofit2.Response
import retrofit2.http.GET

interface ApiService {
    @GET("/predict")
    suspend fun getPredictions(): Response<PredictionResponse>
}