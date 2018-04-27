import sbtsparksubmit.SparkSubmitPlugin.autoImport._

object SparkSubmit {
  lazy val settings =
    SparkSubmitSetting("sparkMNIST",
      Seq(
        "--master", "spark://192.168.0.4:7077",
        "--class", "com.github.saurfang.spark.tsne.examples.MNIST"
      ),
      Seq("hdfs://192.168.0.6:9000/data/tsne/data/mnist.csv")
    )
}
