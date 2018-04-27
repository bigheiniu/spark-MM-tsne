package com.github.saurfang.spark.tsne.examples





import com.github.saurfang.spark.tsne.impl._
import org.apache.spark.mllib.X2PHelper
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory
import org.apache.spark.mllib.feature.PCA

object MNIST {
  private def logger = LoggerFactory.getLogger(MNIST.getClass)

  def main (args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val fileName = args(0)
    val sc = new SparkContext(conf)
    val dataset = sc.textFile(fileName,8)
      .zipWithIndex()
      .map(_._1.split(","))
      .map(x => x.map(_.toDouble))
      .cache()
    val scaledData = dataset.map(x => Vectors.dense(x))
    val matrix = new RowMatrix(scaledData)
    matrix.rows.cache()
//    MultilMap.tsne(matrix,no_maps = 15)
    MultilMap.tsne1(matrix,maxIterations = 500)
    SimpleTSNE.tsne(matrix)
//    MultilMap.tsne(matrix,maxIterations = 2000)

    sc.stop()

//


    //SimpleTSNE.tsne(pcaMatrix, perplexity = 20, maxIterations = 200)
//    BHTSNE.tsne(pcaMatrix, maxIterations = 500, callback = {
//    //LBFGSTSNE.tsne(pcaMatrix, perplexity = 10, maxNumIterations = 500, numCorrections = 10, convergenceTol = 1e-8)
//      case (i, y, loss) =>
//        if(loss.isDefined) logger.info(s"$i iteration finished with loss $loss")
//
//        val os = fs.create(new Path(s".tmp/MNIST/result${"%05d".format(i)}.csv"), true)
//        val writer = new BufferedWriter(new OutputStreamWriter(os))
//        try {
//          (0 until y.rows).foreach {
//            row =>
//              writer.write(labels(row).toString)
//              writer.write(y(row, ::).inner.toArray.mkString(",", ",", "\n"))
//          }
//          if(loss.isDefined) costWriter.write(loss.get + "\n")
//        } finally {
//          writer.close()
//        }
//    })
//    costWriter.close()

//    sc.stop()
//val conf = new SparkConf()
//  .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer").setMaster("local[4]")
//    val sc = new SparkContext(conf)



//    val data1 = sc.textFile("/Users/bigheiniu/course/graduate_pro/spark-tsne-master/data/mnist/mnist.csv")
//    val parsedData1 = data1.map { line =>
//      Vectors.dense(line.split(',').map(_.toDouble))
//    }.cache()
//    val time3 = System.nanoTime()
//    val fuck1 = new RowMatrix(parsedData1)
//    val fuckIndex = new IndexedRowMatrix(fuck1.rows.zipWithIndex.map(x => IndexedRow(x._2,x._1))).toBlockMatrix()
//    val TMatBlock = fuckIndex.transpose
//    val productMatBlock = fuckIndex.multiply(TMatBlock)
//    val productMatRow = productMatBlock.toIndexedRowMatrix().toRowMatrix().rows.first()
//    val time4 = System.nanoTime()
//    val data1 = sc.textFile("data/MNIST/mnist.csv",30)
//      val parsedData1 = data1.map { line =>
//          DenseVector(line.split(',').map(_.toDouble))
//      }.cache()
//    val parseData2 = parsedData1.glom()
//    val fuckdata = parseData2.map(arr => DenseMatrix(arr.map(_.toArray):_*))
//    val cols = fuckdata.first().cols
//    val vecto = DenseVector.fill(cols){0.9}
//    var i = 0
//    while (i < 10) {
//
//      val tim1 = System.nanoTime()
//      fuckdata.map(arr => (arr(*,::) :*  vecto)).count()
//      val tim2 = System.nanoTime()
//      val thu  = parsedData1.map(arr => (arr :* vecto)).count()
//      val tim3 = System.nanoTime()
//      i =i + 1
//      val m = tim2 - tim1
//      val v = tim3 - tim2
//      //the ratio is 6.0
//      //the ratio is 11.0
//      println("the ratio is " + (m / v * 1.0))
//    }

//      val indexSeq = parsedData1.collect().toIndexedSeq
//      sc.stop()
//    // Building the model
//    import java.io._
//    val fuckdata = csvread(new File("/Users/bigheiniu/course/graduate_pro/spark-tsne-master/data/mnist/mnist.csv"), ',')
//    val time5 = System.nanoTime()
//      val len = 0 until fuckdata.rows
//      val i = 10
//      len.par.map(index => fuckdata(10,::) - fuckdata(index,::))
//    val time6 = System.nanoTime()
//      indexSeq.map(arr => Vectors.dense((new DenseVector(arr.toArray) - new DenseVector(indexSeq.apply(10).toArray)).toArray) )
//      val time7 = System.nanoTime()

    // Evaluate model on training examples and compute training error


//    println("fuck" + (1.0 * (tim2-tim1) / (time7 - time6)))
//    println("fuck all " + (1.0 * (time7 - time6)/(time6 - time5)))

      //fuck 3.0052068332108925
      //fuck all 0.707177996366076


  }
}
