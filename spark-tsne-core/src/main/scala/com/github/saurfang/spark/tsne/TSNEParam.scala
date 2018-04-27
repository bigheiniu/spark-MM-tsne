package com.github.saurfang.spark.tsne

case class TSNEParam(
                      early_exaggeration: Int = 50,
                      exaggeration_factor: Double = 4.0,
                      t_momentum: Int = 250,
                      initial_momentum: Double = 0.5,
                      final_momentum: Double = 0.8,
                      eta: Double = 250.0,
                      etw: Double = 100.0,
                      min_gain: Double = 0.01,
                      zoomValue: Double = 4.0,
                      realmin: Double= 1e-12,
                      K:Int = 1,
                      laplace: Double = 0.0
                      )
