package com.github.saurfang.spark.tsne

import java.io.File

import breeze.linalg._
import breeze.stats._
import breeze.numerics.{exp, pow, sqrt}
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.rdd.RDD

object TSNEHelper {
  // p_ij = (p_{i|j} + p_{j|i}) / 2n
  def computeP(p_ji: CoordinateMatrix, n: Int): RDD[(Int, Iterable[(Int, Double)])] = {
    p_ji.entries
      .flatMap(e => Seq(
      ((e.i.toInt, e.j.toInt), e.value),
      ((e.j.toInt, e.i.toInt), e.value)
    ))
      .reduceByKey(_ + _) // p + p'
      .map{case ((i, j), v) => (i, (j, math.max(v / 2 / n, 1e-12))) } // p / 2n
      .groupByKey()
  }

  /**
   * Update Y via gradient dY
   * @param Y current Y
   * @param dY gradient dY
   * @param iY stored y_i - y_{i-1}
   * @param gains adaptive learning rates
   * @param iteration n
   * @param param [[TSNEParam]]
   * @return
   */
  def update(Y: DenseMatrix[Double],
             dY: DenseMatrix[Double],
             iY: DenseMatrix[Double],
             gains: DenseMatrix[Double],
             iteration: Int,
             param: TSNEParam): DenseMatrix[Double] = {
    import param._
    val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
    gains.foreachPair {
      case ((i, j), old_gain) =>
        val new_gain = math.max(min_gain,
          if ((dY(i, j) > 0.0) != (iY(i, j) > 0.0))
            old_gain + 0.2
          else
            old_gain * 0.8
        )
        gains.update(i, j, new_gain)

        val new_iY = momentum * iY(i, j) - eta * new_gain * dY(i, j)
        iY.update(i, j, new_iY)

        Y.update(i, j, Y(i, j) + new_iY) // Y += iY
    }
    val t_Y: DenseVector[Double] = mean(Y(::, *)).t
    val y_sub = Y(*, ::)
    Y := y_sub - t_Y
  }

  def update(Y: DenseMatrix[Double],
             dY: DenseMatrix[Double],
             iY: DenseMatrix[Double],
             iteration: Int,
             param: TSNEParam): DenseMatrix[Double] = {
    import param._
    val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
    iY := momentum * iY - eta * dY
    Y := Y + iY
    val t_Y: DenseVector[Double] = mean(Y(::, *)).t
    val y_sub = Y(*, ::)
    Y := y_sub - t_Y
  }


  def updateMutiMap(
    Y: IndexedSeq[DenseMatrix[Double]],
    nagY: IndexedSeq[DenseMatrix[Double]],
    dY: IndexedSeq[DenseMatrix[Double]],
  iY: IndexedSeq[DenseMatrix[Double]],
    weight: DenseMatrix[Double],
    dW: DenseMatrix[Double],
    iteration: Int,
    Eg_t: IndexedSeq[DenseMatrix[Double]],
    param: TSNEParam
  ): Unit  = {
    import param._
    val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
    val maps = Y.length
    val cols = Y.apply(0).cols
    (0 until maps).par.foreach( map => {

//      nagY.apply(map) := Y.apply(map)  + momentum * iY.apply(map)
      iY.apply(map) := momentum * iY.apply(map) - eta * dY.apply(map)
      Y.apply(map) := Y.apply(map) + iY.apply(map)

      // TODO: 测试出了问题, 暂时不进行调试
      //      val LearnDY = LearnGradient(Eg_t.apply(map), dY.apply(map), eta, realmin)
//      iY.apply(map) := momentum :* iY.apply(map) - eta * dY.apply(map) //LearnDY
//      Y.apply(map) := Y.apply(map) + iY.apply(map)
      val t_Y = mean(Y.apply(map)(::, *)).t
      val y_sub = Y.apply(map)(*, ::)
      Y.apply(map) := y_sub - t_Y
    }
    )
    weight := weight - etw * dW
  }

  def updateMutiMap(
                     Y: IndexedSeq[DenseMatrix[Double]],
                     nagY: IndexedSeq[DenseMatrix[Double]],
                     dY: IndexedSeq[DenseMatrix[Double]],
                     iY: IndexedSeq[DenseMatrix[Double]],
                     weight: DenseMatrix[Double],
                     dW: DenseMatrix[Double],
                     gains: IndexedSeq[DenseMatrix[Double]],
                     iteration: Int,
                     Eg_t: IndexedSeq[DenseMatrix[Double]],
                     param: TSNEParam
                   ): Unit  = {
    import param._
    val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
    val maps = Y.length
    val cols = Y.apply(0).cols
    (0 until maps).par.foreach( map => {
      // Y_(t-1) + gama * Veco_(t-1)
      gains.apply(map).foreachPair {
        case ((i, j), old_gain) =>
          val new_gain = math.max(min_gain,
            if ((dY.apply(map)(i, j) > 0.0) != (iY.apply(map)(i, j) > 0.0))
              old_gain + 0.2
            else
              old_gain * 0.8
          )
          gains.apply(map).update(i, j, new_gain)
          val new_iY = momentum * iY.apply(map)(i, j) - eta * new_gain * dY.apply(map)(i, j)
          iY.apply(map).update(i, j, new_iY)

          Y.apply(map).update(i, j, Y.apply(map)(i, j) + new_iY) // Y += iY
      }
//      nagY.apply(map) := Y.apply(map)  + momentum * iY.apply(map)
//      iY.apply(map) := momentum * iY.apply(map) - eta * dY.apply(map)
//      Y.apply(map) := Y.apply(map) + iY.apply(map)


      // TODO: 测试出了问题, 暂时不进行调试
      //      val LearnDY = LearnGradient(Eg_t.apply(map), dY.apply(map), eta, realmin)
      //      iY.apply(map) := momentum :* iY.apply(map) - eta * dY.apply(map) //LearnDY
      //      Y.apply(map) := Y.apply(map) + iY.apply(map)
      val t_Y = mean(Y.apply(map)(::, *)).t
      val y_sub = Y.apply(map)(*, ::)
      Y.apply(map) := y_sub - t_Y
      nagY.apply(map) := Y.apply(map)
    }
    )
    weight := weight - etw * dW
  }


//  def updateMutiMapOr(
//                     Y: IndexedSeq[DenseMatrix[Double]],
//                     nagY: IndexedSeq[DenseMatrix[Double]],
//                     dY: IndexedSeq[DenseMatrix[Double]],
//                     iY: IndexedSeq[DenseMatrix[Double]],
//                     weight: DenseMatrix[Double],
//                     dW: DenseMatrix[Double],
//                     iteration: Int,
//                     Eg_t: IndexedSeq[DenseMatrix[Double]],
//                     param: TSNEParam
//                   ): Unit  = {
//    import param._
//    val momentum = if (iteration <= t_momentum) initial_momentum else final_momentum
//    val maps = Y.length
//    val cols = Y.apply(0).cols
//    for ( map <- 0 until maps) {
//      // Y_(t-1) + gama * Veco_(t-1)
//      nagY.apply(map) :=  Y.apply(map)
//      // TODO: 测试出了问题, 暂时不进行调试
////      val LearnDY = LearnGradient(Eg_t.apply(map), dY.apply(map),eta, realmin)
//      iY.apply(map) := momentum :* iY.apply(map) - eta * dY.apply(map)
//      Y.apply(map) := Y.apply(map) + iY.apply(map)
//      Y.apply(map) := Y.apply(map) - tile(mean(Y.apply(map).apply(*, ::)), 1, cols)
//    }
//    weight := weight - etw * dW
//  }

  def LearnGradient(Eg_t: DenseMatrix[Double], g_t: DenseMatrix[Double], lr: Double, realMin: Double):DenseMatrix[Double] = {
    Eg_t :=  Eg_t * 0.1 + 0.9 * pow(g_t,2)
    lr * g_t :/ sqrt(Eg_t :+ realMin)
  }




  def GetProportions(weight: DenseMatrix[Double]): DenseMatrix[Double] = {
    val temp = exp(-1.0 * weight)
    val proportions = temp(::, *) :/ sum(temp.apply(*, ::))
    proportions

  }
  def NeiborhoodPreserve(P: (Int, Iterable[(Int,Double)]), Q: DenseVector[Double],K: Int): Double = {
    require(P._2.size >= K,"two many neighbors, please choose a small neighbor counts")
    val PKmax = P._2.toArray.sortBy(-_._2).take(K).map(_._1).toSet
    val QKmax = Q.toArray.zipWithIndex.sortBy(-_._1).take(K).map(_._2).toSet
    PKmax.intersect(QKmax).size / (K * 1.0)
  }

  def NeiborhoodPreserveMatrix(P: Array[(Int, Iterable[(Int,Double)])], Q: DenseMatrix[Double],K: Int): Double = {

    P.zipWithIndex.map(arr => {
      val P_index = P.apply(arr._2)
      val Q_index = Q(arr._2,::).t
      NeiborhoodPreserve(P_index,Q_index,K)
    }).sum

  }
  def ZoomP(P_value: Array[(Int,Iterable[(Int, Double)])], zoomValue: Double): Array[(Int,Iterable[(Int, Double)])] = {
    P_value.map(arr => (arr._1,arr._2.map(th => (th._1,th._2 * zoomValue))))
//    P_value.map(arr => (arr._1, arr._2 * zoomValue))
  }

  def ZoomP(P_value: Iterable[(Int, Double)], zoomValue: Double): Iterable[(Int, Double)] = {
        P_value.map(arr => (arr._1, arr._2 * zoomValue))
  }

  def FindNan(denseMatrix: DenseMatrix[Double]): Boolean = {
    denseMatrix.valuesIterator.exists(_.isNaN)
  }

  def FindNanVect(denseVector: DenseVector[Double]): Boolean = {
    denseVector.valuesIterator.exists(_.isNaN)
  }
}
