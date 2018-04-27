
package com.github.saurfang.spark.tsne.impl

import java.io.{BufferedWriter, File, FileWriter}

import breeze.linalg._
import breeze.stats.distributions.Rand
import com.github.saurfang.spark.tsne._
import org.apache.spark.HashPartitioner
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.util.Random

object MMatrix {
  private def logger = LoggerFactory.getLogger(SimpleTSNE.getClass)

  def tsne(
            input: RowMatrix,
            noDims: Int = 2,
            no_maps: Int = 15,
            maxIterations: Int = 1000,
            perplexity: Double = 30,
            callback: (Int, DenseMatrix[Double], Option[Double]) => Unit = {case _ => },
            seed: Long = Random.nextLong()): DenseMatrix[Double] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logger.warn("Input is not persisted and performance could be bad")
    }

    Rand.generator.setSeed(seed)

    val tsneParam = TSNEParam()
    import tsneParam._

    val n = input.numRows().toInt
    val Y = for (i <- 0 until no_maps)  yield DenseMatrix.rand(n,noDims, Rand.gaussian(0, 1))
    val nagY = for (i <- 0 until no_maps)  yield DenseMatrix.rand(n,noDims, Rand.gaussian(0, 1))
    val Eg_t = for (i <- 0 until no_maps)  yield DenseMatrix.zeros[Double](n,noDims)
//    val Y: DenseMatrix[Double] = DenseMatrix.rand(n, noDims, Rand.gaussian(0, 1))
    val iY = for (i <- 0 until no_maps)  yield DenseMatrix.zeros[Double](n,noDims)
    val gains = for(i <- 0 until no_maps) yield DenseMatrix.ones[Double](n, noDims)

    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    val P1 = TSNEHelper.computeP(p_ji, n)
    val P_constant = P1.partitionBy(new HashPartitioner(120)).glom().cache()
    val P_big = P_constant.map(arr => TSNEHelper.ZoomP(arr,zoomValue))

    val weights = DenseMatrix.fill(n,no_maps){1.0 / no_maps}
    val L_cons = P_constant.map(data => TSNEGradientMatrix.GetL(data,n)).cache()
    //@CACHE
    val L_big = P_big.map(data => TSNEGradientMatrix.GetL(data,n))

    var iteration = 1
    while(iteration <= maxIterations) {
      val P= if(iteration <= early_exaggeration) P_big else P_constant
      val L = if(iteration <= early_exaggeration) L_big else L_cons

      val bcY = P.context.broadcast(nagY)
      val proportions = TSNEHelper.GetProportions(weights)
      val bcPro = P.context.broadcast(proportions)
      val result = P.map{ arr => TSNEGradientMatrix.Numerator_QQ_QZ(bcPro.value,noDims,no_maps,n,bcY.value, arr.map(_._1): _*) }.cache()
//      val result1 = P.map(arr => TSNEGradient.computeNumerator(bcY.value.apply(0),arr.map(_._1):_*)).cache()
      val Z =
        result.map(_._2).treeAggregate(0.0)(
          seqOp = (c, v) => c + sum(v),
          combOp = (c1, c2) => c1 + c2
        )
//      val QZ = result.map(_._2)
//      val QQ = result.map(arr => arr._1.map(_._2))
//      val num = result.map(arr => arr._1.map(_._1))
//     num.zip(QZ).foreach(arr => println(arr._1.apply(0) - arr._2))

      val target = P.zip(result).zip(L).mapPartitions(par => par.map(
        arr => {
          val QZ = arr._1._2._2
          val QQ = arr._1._2._1.map(_._2)
          val num = arr._1._2._1.map(_._1)
          val data = arr._1._1
          val Q = QZ :/ Z
          val laplace = 0.0
          val l = arr._2
          TSNEGradientMatrix.compute(data,bcY.value,num,QQ,Z,n,no_maps,noDims,1 < 2,Q,QZ,bcPro.value,l,laplace)
        }
      )).cache()
      val dY1 = (0 until no_maps).map(_ => DenseMatrix.zeros[Double](n,noDims))
      for(map <- 0 until no_maps)  yield target.flatMap(_._2.apply(map)).collect().foreach( arr => dY1.apply(map)(arr._1,::) := arr._2.t)

      val dCdW = DenseMatrix.zeros[Double](n,no_maps)
      target.flatMap(_._3).collect().foreach(
        arr => dCdW(arr._1,::) := arr._2.t
      )

      //      TSNEHelper.update(Y.apply(0), dY1, iY.apply(0), gains, iteration, tsneParam)



//      val result = P.zip(numerator).map(arr => (TSNEGradient.compute1(arr._1._1,bcY.value,arr._2,bcNumerator.value,iteration<=early_exaggeration),arr._1._2)).cache()
      //(Double,Array[(Int,DenseVector[Double])])
//      val dY1 = DenseMatrix.zeros[Double](n,noDims)
//      val
//      val result = P.mapPartitions { th => th.map(arr => TSNEGradientMatrix.Numerator_QQ_QZ(proportions, noDims, no_maps, n, bcY.value,arr._1.map(_._1):_*)) }.cache()
//      val loss1 = result.mapPartitions(arr => arr.map(_._1._1)).sum()
//      result.collect().flatMap(_._1._2).foreach(
//        arr => dY1(arr._1,::) := arr._2
//      )
      //      val target = result.collect.flatMap(_._1._2).sortBy(_._1)
      //      val dY1 = DenseMatrix(target.map(_._2.toArray):_*)

      //        .valuesIterator.exists(_ == 0)) println("fuck the bitches all ")
      if (iteration % 100 ==  0) {

        val newbcY = P.context.broadcast(Y)

        val result1 = P.map{ arr => TSNEGradientMatrix.Numerator_QQ_QZ(bcPro.value,noDims,no_maps,n,newbcY.value, arr.map(_._1): _*) }.cache()
        val QZ = result1.map(_._2)
        val Q = QZ.map(arr => arr :/ Z)
        val neighbor = P.zip(Q).map(arr => TSNEHelper.NeiborhoodPreserveMatrix(arr._1, arr._2, K)).sum() / (n * 1.0)
        newbcY.unpersist(blocking = false)
        result1.unpersist(blocking = false)
        val file1 = new File(s"logs/neighbor_2_NagL.txt")
        val bw = new BufferedWriter(new FileWriter(file1,true))
        bw.write(s"the${iteration} is :" + neighbor.toString + "\n")
        bw.close()
      }
      System.out.println(s"the iteration is ${iteration}")

      bcY.destroy()
      result.unpersist(blocking = false)
      target.unpersist(blocking = false)
//      TSNEHelper.updateMutiMap(Y, nagY,dY1, iY,weights,dCdW,iteration,Eg_t,tsneParam)
//      TSNEHelper.updateMutiMap(Y,dY1,iY,gains,iteration,tsneParam )


//      logger.debug(s"Iteration $iteration finished with $loss1")
//      callback(iteration, Y.copy, Some(loss1))
      iteration += 1
    }
    L_cons.unpersist(blocking = false)
    P_constant.unpersist(blocking = false)

    Y.apply(0)
  }
}