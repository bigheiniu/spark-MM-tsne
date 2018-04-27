package example

import com.github.saurfang.spark.tsne.impl._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory

object MNIST {
  private def logger = LoggerFactory.getLogger(MNIST.getClass)

  def main (args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val fileName = args(0)
    logger.debug(args(0))
    val sc = new SparkContext(conf)
    val dataset = sc.textFile(fileName,32)
      .zipWithIndex()
      .map(_._1.split(","))
      .map(x => x.map(_.toDouble))
      .cache()
    val scaledData = dataset.map(x => Vectors.dense(x))
    val matrix = new RowMatrix(scaledData)
    matrix.rows.cache()
    MultilMap.tsne1(matrix,maxIterations = 1000)
    SimpleTSNE.tsne(matrix,maxIterations = 1000)

    sc.stop()

  }
}
