package com.github.saurfang.spark.tsne.impl

import java.io.{BufferedWriter, File, FileWriter}

import breeze.linalg._
import breeze.stats.distributions.Rand
import com.github.saurfang.spark.tsne.{TSNEGradient, TSNEHelper, TSNEParam, X2P}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object SimpleTSNE {
  private def logger = LoggerFactory.getLogger(SimpleTSNE.getClass)

  def tsne(
            input: RowMatrix,
            noDims: Int = 2,
            maxIterations: Int = 1000,
            perplexity: Double = 30,
            callback: (Int, DenseMatrix[Double], Option[Double]) => Unit = {case _ => },
            seed: Long = Random.nextLong()): DenseMatrix[Double] = {
    if(input.rows.getStorageLevel == StorageLevel.NONE) {
      logger.warn("Input is not persisted and performance could be bad")
    }
    val time1 = System.currentTimeMillis();
    Rand.generator.setSeed(seed)

    val tsneParam = TSNEParam()
    import tsneParam._

    val n = input.numRows().toInt
    val Y: DenseMatrix[Double] = DenseMatrix.rand(n, noDims, Rand.gaussian(0, 1))
    val iY = DenseMatrix.zeros[Double](n, noDims)
    val gains = DenseMatrix.ones[Double](n, noDims)

    // approximate p_{j|i}
    val p_ji = X2P(input, 1e-5, perplexity)
    val P = TSNEHelper.computeP(p_ji, n).glom().cache()

    var iteration = 1
    val fileSave = new ArrayBuffer[String]();
    while(iteration <= maxIterations) {
      val bcY = P.context.broadcast(Y)
/**/
      val numerator = P.map{ arr => TSNEGradient.computeNumerator(bcY.value, arr.map(_._1): _*) }.cache()
      val bcNumerator = P.context.broadcast({
        numerator.treeAggregate(0.0)(seqOp = (x, v) => x + sum(v), combOp = _ + _)
      })

      val (dY, loss) = P.zip(numerator).treeAggregate((DenseMatrix.zeros[Double](n, noDims), 0.0))(
        seqOp = (c, v) => {
          // c: (grad, loss), v: (Array[(i, Iterable(j, Distance))], numerator)
          val l = TSNEGradient.compute(v._1, bcY.value, v._2, bcNumerator.value, c._1, iteration <= early_exaggeration)
          (c._1, c._2 + l)
        },
        combOp = (c1, c2) => {
          // c: (grad, loss)
          (c1._1 + c2._1, c1._2 + c2._2)
        })
      if (iteration % 100 == 0) {
        val Q = numerator.map(arr => arr / bcNumerator.value)
        val neighbor = P.zip(Q).map(arr => TSNEHelper.NeiborhoodPreserveMatrix(arr._1, arr._2, K)).sum() / (n * 1.0)
        fileSave.append(s"the${iteration} is : ${neighbor} \n")
        logger.debug(s"the single Maps ${iteration} is : ${neighbor}")
      }

      bcY.destroy()
      bcNumerator.destroy()
      numerator.unpersist()

      TSNEHelper.update(Y, dY, iY, gains, iteration, tsneParam)
      callback(iteration, Y.copy, Some(loss))
      iteration += 1
    }
    val time2 = System.currentTimeMillis();
    val time = (time2 - time1) / 1e9d
    fileSave.append(s"Time is ${time} seconds")
    P.sparkContext.parallelize(fileSave.toSeq).saveAsTextFile("hdfs://192.168.0.6:9000/data/tsne/result/singleMap")
    Y
  }
}


