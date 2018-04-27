
package com.github.saurfang.spark.tsne.impl


import breeze.linalg._
import breeze.numerics.pow
import breeze.stats.distributions.Rand
import com.github.saurfang.spark.tsne._
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object MultilMap {
  private def logger = LoggerFactory.getLogger(SimpleTSNE.getClass)

  def tsne1(input: RowMatrix,
            noDims: Int = 2,
            no_maps: Int = 15,
            maxIterations: Int = 2000,
            perplexity: Double = 30,
            callback: (Int, DenseMatrix[Double], Option[Double]) => Unit = {case _ => },
            seed: Long = Random.nextLong()): DenseMatrix[Double] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logger.warn("Input is not persisted and performance could be bad")
    }

    Rand.generator.setSeed(seed)
    val time1 = System.currentTimeMillis();
    val tsneParam = TSNEParam()
    import tsneParam._

    val n = input.numRows().toInt
    val Y = for (i <- 0 until no_maps)  yield DenseMatrix.rand(n,noDims, Rand.gaussian(0, 1))
    val nagY = for (i <- 0 until no_maps)  yield DenseMatrix.rand(n,noDims, Rand.gaussian(0, 1))
    val iY = for (i <- 0 until no_maps)  yield DenseMatrix.zeros[Double](n,noDims)
    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    val P_constant = TSNEHelper.computeP(p_ji, n).cache()
    val P_big = P_constant.map(arr => (arr._1, TSNEHelper.ZoomP(arr._2,zoomValue)))

    val weights = DenseMatrix.fill(n,no_maps){1.0 / no_maps}
    val L_cons = P_constant.map(data => TSNEGradientVect.GetL(data,n)).cache()
    //@CACHE
    val L_big = P_big.map(data => TSNEGradientVect.GetL(data,n)).cache()

    var iteration = 1
    val fileSave = new ArrayBuffer[String]();
    while(iteration <= maxIterations) {
      val P = if(iteration <= early_exaggeration) P_big else P_constant
      val L = if(iteration <= early_exaggeration) L_big else L_cons

      val bcY = P.context.broadcast(Y)
      val sumY: IndexedSeq[DenseVector[Double]] = Y.map(arr => sum(pow(arr, 2).apply(*, ::)))
      val bcSumY = P.context.broadcast(sumY)

      // proportions 矩阵传播
      val proportions = TSNEHelper.GetProportions(weights)
      //      val proportions = DenseMatrix.fill(n,no_maps){1.0} /*TSNEHelper.GetProportions(weights)*/
      val bcPro = P.context.broadcast(proportions)

      val result = P.map{ arr => TSNEGradientVect.Numerator_QQ_QZ(bcPro.value,noDims,no_maps,n,arr._1, bcY.value,bcSumY.value) }.cache()

      // TODO: 针对 depth 可以根据 partition 进行一系列的优化
      val Z =
        result.map(_._2).treeAggregate(0.0)(
          seqOp = (c, v) => c + sum(v),
          combOp = (c1, c2) => c1 + c2
        )

      val QZ = result.map(_._2)
      val QQ = result.map(_._1.map(_._2))
      val num = result.map(_._1.map(_._1))

      val Q = QZ.map(arr => TSNEGradientVect.GetQ(arr,Z))

      val PQ = Q.zip(P).map(arr => TSNEGradientVect.GetPQ(arr._2,arr._1))

      val numerQQIndex = result.map(arr => (arr._1,arr._3))

      val numerIndex = numerQQIndex.map(arr => (arr._1.map(_._1), arr._2))

      val dCdD = PQ.zip(numerQQIndex).zip(QZ).map(arr => {
        val PQ_i = arr._1._1
        val numerQQIndex_i = arr._1._2
        val QZ_i = arr._2
        TSNEGradientVect.ComputedCdD(PQ_i,numerQQIndex_i,QZ_i)
      })


      val dCdY = dCdD.map(arr => TSNEGradientVect.ComputedCdY(arr,bcY.value,laplace,n,no_maps,noDims))
      val dY1 = (0 until no_maps).map(_ => DenseMatrix.zeros[Double](n,noDims))
      dCdY.collect().par.foreach(arr => {
        (0 until no_maps).par.foreach(
          mapIndex => dY1.apply(mapIndex)(arr._2,::) := arr._1.apply(mapIndex).t
        )
      })

      val tmpPQ = QZ.zip(PQ).map(arr => TSNEGradientVect.GettempPQ(arr._1,arr._2))
      val dCdW = numerIndex.zip(tmpPQ).zip(L).map(arr => TSNEGradientVect.ComputdCdW(arr._1._2,bcPro.value,arr._1._1._1,arr._1._1._2,arr._2,laplace))
      val dW1 = DenseMatrix.zeros[Double](n,no_maps)
      dCdW.collect().par.foreach(arr => {
        dW1(arr._2,::) := arr._1.t
      })

      bcSumY.unpersist()
      bcY.unpersist()
      if (iteration % 100 ==  0) {

        val newbcY = P.context.broadcast(Y)
        val sumY = Y.map(arr => sum(pow(arr, 2).apply(*, ::)))
        val bcSumY = P.context.broadcast(sumY)
        val result1 = P.map{ arr => TSNEGradientVect.Numerator_QQ_QZ(bcPro.value,noDims,no_maps,n,arr._1,newbcY.value, bcSumY.value) }.cache()
        val QZ = result1.map(_._2)
        val Q = QZ.map(arr => arr :/ Z)
        val neighbor = P.zip(Q).map(arr => TSNEHelper.NeiborhoodPreserve(arr._1, arr._2, K)).sum() / (n * 1.0)
        newbcY.unpersist(blocking = false)
        result1.unpersist(blocking = false)
        fileSave.append(s"the${iteration} is :" + neighbor.toString + "\n")
        logger.debug(s"the Multiple Maps ${iteration} is :" + neighbor.toString )

      }
      result.unpersist(blocking = false)
      TSNEHelper.updateMutiMap(Y,dY1, iY, weights, dW1, iteration, tsneParam)
      iteration += 1
    }
    val time2 = System.currentTimeMillis()
    val time = ( time2 - time1) / 1e9d
    fileSave.append(s"All time Multiple map ${time}")
    P_constant.sparkContext.parallelize(fileSave.toSeq).saveAsTextFile("hdfs://192.168.0.6:9000/data/tsne/result/MultipleMap")
    L_cons.unpersist(blocking = false)
    P_constant.unpersist(blocking = false)

    Y.apply(0)
  }
}