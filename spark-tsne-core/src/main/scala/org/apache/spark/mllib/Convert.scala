package org.apache.spark.mllib

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, DenseMatrix => BDM}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
object Convert {


  def RDDIndeSeq2BDM(rdd: RDD[(IndexedSeq[BDV[Double]],Int)], maps: Int): IndexedSeq[BDM[Double]] = {
    val temp = rdd.collect().sortBy(_._2).map(_._1)
    val bdm = for (i <- 0 until maps) yield {
      val dCdY_m = temp.map(arr => arr.apply(i))
      BDM(dCdY_m.map(_.toArray):_*)
    }
    bdm
  }
  def RDD2BDM(rdd: RDD[(BDV[Double],Int)]): BDM[Double] = {
    val array = rdd.collect().sortBy(_._2).map(_._1)
    BDM(array.map(_.toArray):_*)
  }

  def RDDIndeSeqMatrix2BDM(rdd: RDD[(IndexedSeq[BDM[Double]],Seq[Int])], maps: Int): IndexedSeq[BDM[Double]] = {
    val temp = rdd.collect().map(_._1)
    val bdm = for (i <- 0 until maps) yield {
      val dCdY_m = temp.map(arr => arr.apply(i))
      BDM.vertcat(dCdY_m:_*)
    }
    bdm
  }

  def RDDMatrix2BDM(rdd: RDD[(BDM[Double],Seq[Int])]): BDM[Double] = {
    val array = rdd.collect().map(_._1)
    BDM.vertcat(array:_*)
  }


  def Vect2Matrix(array: Array[BDV[Double]]): BDM[Double] = {
    BDM(array.map(_.toArray):_*)
  }
}
