package com.github.saurfang.spark.tsne

import breeze.numerics.pow
import breeze.linalg._
import org.apache.spark.mllib.Convert

import scala.tools.nsc.transform.patmat.Debugging


object TSNEGradientMatrix {
  val tsneParam = TSNEParam()
  import tsneParam.realmin
  /**
    * Compute the numerator: (1 + ||yi - yj|| ^ 2) ^ -1
    *
    * @param Y     降维后的矩阵
    * @param idx yi 的索引
    * @return 返回(1 + ||y_index - y_j|| ^ 2) ^ -1, 大小为 1 * n 的向量
    **/
  def CompuNumerator(Y: DenseMatrix[Double], idx: Int *): DenseMatrix[Double] = {

    val sumY = sum(pow(Y, 2).apply(*, ::)) // n * 1
    val subY = Y(idx, ::).toDenseMatrix // k * 1
    val y1: DenseMatrix[Double] = Y * (-2.0 :* subY.t) // n * k
    val num: DenseMatrix[Double] = (y1(::, *) + sumY).t // k * n
    num := 1.0 :/ (1.0 :+ (num(::, *) + sumY(idx).toDenseVector)) // k * n
    idx.indices.foreach(i => num.update(i, idx(i), 0.0)) // num(i, i) = 0
    num
  }

  /**
    * compute QQ, QZ matrix, QQ_i_m = π_i * π.t * numerator_m; QZ = QZ + QQ(:,:,m)
    *
    * @param Y           降维后矩阵
    * @param proportions 图的权重矩阵
    * @param no_dims     降维后的维度
    * @param no_maps     图的数量
    * @param idx         i
    */

  def Numerator_QQ_QZ(proportions: DenseMatrix[Double],
                      no_dims: Int,
                      no_maps: Int,
                      rows: Int,
                      Y: IndexedSeq[DenseMatrix[Double]],
                      idx: Int *
                     ):
  (IndexedSeq[(DenseMatrix[Double], DenseMatrix[Double])], DenseMatrix[Double]) = {
    val n = rows
    val numerQQ = Y.zipWithIndex.map{
      case (ym ,map) => {
        val numer = CompuNumerator(ym, idx:_*)
        val fuck1 = proportions(idx,::).toDenseMatrix
        val fuck = fuck1(::,map).toDenseMatrix
        val QQ  = numer :* (fuck.t * proportions (::, map).t)
        (numer, QQ)
      }
    }

    // TODO: 可以多线程实现
    val QZ = numerQQ.map(arr => arr._2).fold(DenseMatrix.fill(idx.length,rows){realmin})(_ + _)
    (numerQQ, QZ)
  }



  // PQ = Q - P
  def GetPQ_lossB4(P: Array[(Int, Iterable[(Int,Double)])], Q: DenseMatrix[Double]):(Double) = {
    P.zipWithIndex.flatMap {
      case ((_, itr), i) =>
        itr.map{
          case (j, p) =>
            val qij = Q(i, j)
            val l = p * math.log(p / qij)
            Q.update(i, j,  qij - p)
            if(l.isNaN) 0.0 else l
        }
    }.sum
  }

  def GettempPQ(QZ: DenseMatrix[Double], PQ: DenseMatrix[Double]): DenseMatrix[Double] = {
    (1.0 :/ QZ) :* PQ
  }


  def GetQ(QZ: DenseMatrix[Double], Z: Double): DenseMatrix[Double] = {
    val Q = QZ :/ Z
    Q.foreachPair{case ((i, j), v) => Q.update(i, j, math.max(v, 1e-12))}
    Q
  }

  def GetL(P: Array[(Int, Iterable[(Int, Double)])],n : Int): DenseMatrix[Double] = {
    val value = P.map(arr => arr._2.aggregate(0.0)(
      seqop = (c,v) => c + v._2,
      combop = _ + _
    ))

    val length = P.length
    val LMatrix = DenseMatrix.fill(length,n){realmin}
    value.zip(P).zipWithIndex.foreach(
      arr => {
        val row = arr._2
        val cols = arr._1._2._1
        val value = arr._1._1
        val Pvect = arr._1._2._2
        LMatrix.update(row,cols,value)
        Pvect.foreach{
          case (j, v1) => {
            val old = LMatrix(row,j)
            LMatrix.update(row,j,old - v1)
          }
        }
      }
    )
    //TODO: LMatrix 可以设计成 sparseMatrix
   LMatrix
  }
  /**
    * 计算 dCdD
    *
    * @param numerQQ_index (numer, QQ)
    * @param QZ      QZ matrix
    * @param laplace laplace 系数
    * @return 返回 dCdW, dCdD
    **/
  /**
    * for m=1:no_maps
            dCdD(:,:,m) =(1-laplace) * (QQ(:,:,m) ./ QZ) .* -PQ .* num(:,:,m);%代价函数对距离的更新，加拉普拉斯算子
        end
    */

  // PQ = P - Q; P 是类似于稀疏矩阵的设计, 而 Q 直接是 densematrix 即可
  def ComputedCdD(PQ: DenseMatrix[Double],
                  numer: DenseMatrix[Double],
                  QQ: DenseMatrix[Double],
                  QZ: DenseMatrix[Double],
                  laplace: Double): DenseMatrix[Double]= {
    // 需要重新计算
    // test
    //      (numerQQ._1.map(arr =>  DenseVector.zeros[Double](n) + 1.0), numerQQ._2)

    (1.0 - laplace) * (QQ :/ QZ) * -1.0 :* PQ :* numer

  }


  // the difference between the list and indexedseq
  def ComputedCdY(dCdD: DenseMatrix[Double], Y: DenseMatrix[Double], n: Int, no_maps: Int, no_dims: Int, index: Int *): Array[(Int,DenseVector[Double])] = {

    index.zipWithIndex.par.map(arr => {
      val row = arr._1 // Y local matrix's index -> i
      val idx = arr._2 // dCdD distributed matrix's index ->
      (row, sum((tile(4.0 * dCdD(idx,::).t, 1, no_dims) :* (tile(Y(row, ::), 1, n) - Y)).apply(::,*)).t)
    }).toArray
  }


  /**
    * PQ = Q - P;
        tmp = (1 ./ QZ) .* PQ;
        for m=1:no_maps
            %dCdP(i,m) = sum(bsxfun(@times, proportions(i,m), num(i,:,m) .* tmp), 1)' ;
            dCdP(:,m) = sum(bsxfun(@times, proportions(:,m), num(:,:,m) .* tmp), 1)' ;
        end
        % n * m
        dCdP = 2 * dCdP + laplace *(L*proportions);
    */

  def ComputdCdW(tmpPQ: DenseMatrix[Double],
                 proportions: DenseMatrix[Double],
                 numer: IndexedSeq[DenseMatrix[Double]],
                 L: DenseMatrix[Double],
                 laplace: Double,
                 no_maps: Int,
                 idx: Int *
                ): Array[(Int,DenseVector[Double])]= {

    val proIndex = proportions(idx,::).toDenseMatrix
    // dCdP 是 m * no_dims 矩阵
    val dCdP = 2.0 * Convert.Vect2Matrix(numer.zipWithIndex.map(arr => ComputedCdP(proIndex(::, arr._2),arr._1,tmpPQ)).toArray).t + laplace * ( L * proportions)
    //  no_maps个元素
    //% dCdW_i = pro_i :* (tile(sum(dCdP_i :* pro_i).apply(*,::),1,no_dims) - dCdP_i)
    val dCdW_matrix: DenseMatrix[Double] = proIndex :* (tile(sum((dCdP :* proIndex).apply(*,::)), 1, no_maps) - dCdP)
    idx.zipWithIndex.map(arr => {
      val idx = arr._2
      val row = arr._1
      (row,dCdW_matrix(idx,::).t)
    }).toArray
  }

  // 返回 m * 1
  private  def ComputedCdP(proportions: DenseVector[Double],
                           numer: DenseMatrix[Double],
                           tmp: DenseMatrix[Double]): DenseVector[Double] = {
    val cols = numer.cols
    sum((tile(proportions,1,cols) :* numer :* tmp).apply(*,::))
  }

  def ZoomP(P: Array[(Int,Iterable[(Int,Double)])], zoomValue: Double, exaggeration: Boolean): Array[(Int,Iterable[(Int,Double)])] = {
    if (exaggeration)
      P.map(P => (P._1,P._2.map(arr => (arr._1, arr._2 * zoomValue))))
    else
      P
  }

  /**
    * loss函数
    *
    * @param Q Q矩阵
    * @param P P 矩阵
    **/

  def Compute_loss(Q: DenseVector[Double], P: (Int, Iterable[(Int, Double)]), propotions: DenseMatrix[Double], L: DenseVector[Double], laplace: Double): Double = {
    val PQ_loss = DenseVector.zeros[Double](Q.length)
    val index = P._1

    val loss_before = P._2.aggregate(0.0)(
      seqop = (c, v) => v match
      {
        case ( j, value) => {
          val loss = value * math.log(math.max(value, realmin) / math.max(Q(j), realmin))
          if (loss.isNaN) 0.0 else loss
        }
      },
      combop = _ + _
    )

    val loss_end = laplace * sum(propotions.t * L * propotions(index,::))
    loss_end + loss_before
  }

//  def Compute_loss(proportions: DenseMatrix[Double], L: DenseMatrix[Double], laplace: Double): Double = {
//    laplace * sum(proportions.t * L * proportions)
//  }

  def compute(
               data: Array[(Int, Iterable[(Int, Double)])],
               Y: IndexedSeq[DenseMatrix[Double]],
               num: IndexedSeq[DenseMatrix[Double]],QQ: IndexedSeq[DenseMatrix[Double]],
               totalNum: Double,n: Int, no_maps: Int, no_dims: Int,
               exaggeration: Boolean,
               Q:DenseMatrix[Double],QZ: DenseMatrix[Double],proportions: DenseMatrix[Double],L: DenseMatrix[Double],laplace: Double): (Double,IndexedSeq[Array[(Int, DenseVector[Double])]],Array[(Int, DenseVector[Double])])= {
    // q = (1 + ||Y_i - Y_j||^2)^-1 / sum(1 + ||Y_k - Y_l||^2)^-1

    // q = q - p
    val lossB4 = GetPQ_lossB4(data,Q)
    val PQ = Q
    val index = data.map(_._1)
    // l_sum = [0 0 ... sum(l) ... 0]

    // 现在 Q 代表 PQ
    val dCdY = (0 until no_maps).map( map => {
      val Y_m = Y.apply(map)
      val numer_m = num.apply(map)
      val QQ_m = QQ.apply(map)

       ComputedCdY(
        ComputedCdD(PQ,numer_m, QQ_m,QZ,laplace),Y_m,n,no_maps,no_dims,index:_*)
    })
    val tmp = (1.0 :/ QZ ) :* PQ
    val dCdW = ComputdCdW(tmp,proportions,num, L, laplace, no_maps,index:_*)
    if(TSNEHelper.FindNanVect(dCdW(1)._2)) {
      println("bitch")
    }
    // l = [ (p_ij - q_ij) * (1 + ||Y_i - Y_j||^2)^-1 ]
    (lossB4,dCdY,dCdW)
  }

}
