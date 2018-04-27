package com.github.saurfang.spark.tsne

import breeze.numerics.pow
import breeze.linalg._


object TSNEGradientVect {
 val tsneParam = TSNEParam()
 import tsneParam.realmin



 /**
   * Compute the numerator: (1 + ||yi - yj|| ^ 2) ^ -1
   *
   * @param Y     降维后的矩阵
   * @param index yi 的索引
   * @return 返回(1 + ||y_index - y_j|| ^ 2) ^ -1, 大小为 n * 1 的向量
   **/
 def CompuNumerator(Y: DenseMatrix[Double], index: Int, sumY: DenseVector[Double]): DenseVector[Double] = {

//  val sumY = sum(pow(Y, 2).apply(*, ::)) // n * 1; y_i ^ 2 + y_j ^ 2
  val subY = Y(index, ::) // 1 * ns
  val y1: DenseVector[Double] = Y * (-2.0 :* subY.t) // n * 1;  -2 * y_i * y_j
  val num: DenseVector[Double] = y1 + sumY // n * 1 => y_j ^ 2 - 2 * y_i * y_j
  num := 1.0 :/ (1.0 :+ (num + sumY(index))) // n * 1 => 1 / (1 + y_j ^ 2 - 2 * y_i * y_j + y_i ^ 2 )
  num.update(index,0.0)  // num(i, i) = 0
  num
 }

 /**
   * compute QQ, QZ matrix, QQ_i_m = π_i * π.t * numerator_m; QZ = QZ + QQ(:,:,m)
   *
   * @param Y           降维后矩阵
   * @param proportions 图的权重矩阵
   * @param no_dims     降维后的维度
   * @param no_maps     图的
   * @param idx         i
   */

 def Numerator_QQ_QZ(proportions: DenseMatrix[Double],
                     no_dims: Int,
                     no_maps: Int,
                     rows: Int,
                     idx: Int,
                     Y: IndexedSeq[DenseMatrix[Double]],
                     sumY: IndexedSeq[DenseVector[Double]]
                    ):
 (IndexedSeq[(DenseVector[Double], DenseVector[Double])], DenseVector[Double], Int) = {
  val n = rows
  val maps = Y.length
  val numerQQ = (0 until maps).par.map{
   mapIndex => {
    // sumY.apply(mapIndex) => 特定一张图 Y 的 sum
    val numer = CompuNumerator(Y.apply(mapIndex), idx,sumY.apply(mapIndex))
    val QQ: DenseVector[Double] = numer :* (proportions(idx,mapIndex) * proportions(::, mapIndex)) // QQ 为 n * 1
    (numer, QQ)
   }
  }.toIndexedSeq

  // TODO: 可以多线程实现
  val QZ = numerQQ.map(arr => arr._2).fold(DenseVector.fill(n){realmin})(_ + _) // QZ = sum(QQ); 根据 map 的数量进行求和
  (numerQQ, QZ, idx)
 }


 /**
   * 计算 dCdD
   *
   * @return 返回 dCdW, dCdD
   **/
 /**
   * for m=1:no_maps
            dCdD(:,:,m) =(1-laplace) * (QQ(:,:,m) ./ QZ) .* -PQ .* num(:,:,m);%代价函数对距离的更新，加拉普拉斯算子
        end
   */

 // PQ = P - Q; P 是类似于稀疏矩阵的设计, 而 Q 直接是 densematrix 即可
 def ComputedCdD(PQ: DenseVector[Double],
                 numerQQIndex: (IndexedSeq[(DenseVector[Double], DenseVector[Double])], Int),
                 QZ: DenseVector[Double]
                ): (IndexedSeq[DenseVector[Double]], Int) = {


  def helpdCdD(num_m: DenseVector[Double], QQ_m: DenseVector[Double], PQ: DenseVector[Double], QZ: DenseVector[Double]): DenseVector[Double] = {
    (QQ_m :/ QZ) * -1.0 :* PQ :* num_m
  }

  //      (numerQQ._1.map(arr =>  DenseVector.zeros[Double](n) + 1.0), numerQQ._2)
  val index = numerQQIndex._2
  // bug fix: QQ 和 num 位置有问题 => 已经修复
  (numerQQIndex._1.map(arr => helpdCdD(arr._1, arr._2, PQ, QZ)), index)
 }


 // the difference between the list and indexedseq
 def ComputedCdY(dCdD: (IndexedSeq[DenseVector[Double]], Int),
                 Y: IndexedSeq[DenseMatrix[Double]],laplace: Double, n: Int, no_maps: Int, no_dims: Int) : (IndexedSeq[DenseVector[Double]],Int) = {

  def helpdCdY(dCdD_m: DenseVector[Double], Y_m: DenseMatrix[Double],index: Int): DenseVector[Double] = {
   // tile(Y_m(index, ::), 1, n) - Y_m  =>   y_i(m) - y_j(m) 构成的一个矩阵
   2.0 * (1.0 - laplace) * sum((tile(dCdD_m, 1, no_dims) :* (tile(Y_m(index, ::), 1, n) - Y_m)).apply(::, *)).t
  }

  val index = dCdD._2
  val dCdY = (0 until no_maps).par.map( mapIndex => {
   val dCdD_m: DenseVector[Double] = dCdD._1.apply(mapIndex) // n * 1
   val Y_m: DenseMatrix[Double] = Y.apply(mapIndex) // n * m
   helpdCdY(dCdD_m, Y_m, index) // 1 * m
  }
  ).toIndexedSeq
  (dCdY,index)
 }  // test 样例仅仅检查一个图的正确率 ***** very important


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

 def ComputdCdW(tmpPQ: DenseVector[Double],
                proportions: DenseMatrix[Double],
                numer: IndexedSeq[DenseVector[Double]],
                idx: Int,
                L: DenseVector[Double],
                laplace: Double
               ): (DenseVector[Double],Int)= {

  //for m=1:no_maps
  //dCdP(:,m) = sum(bsxfun(@times, proportions(:,m), num(:,:,m) .* tmp), 1)';
  //end
  //dCdP = 2 * dCdP

  //  dCdW = proportions .* bsxfun(@minus, sum(dCdP .* proportions, 2), dCdP);
  val dCdP = 2.0 * DenseVector(numer.zipWithIndex.map(arr => ComputedCdP(proportions(::, arr._2),arr._1,tmpPQ)).toArray) + laplace * (proportions.t * L)
  //  no_maps个元素
  //dCdW = proportions .* bsxfun(@minus, sum(dCdP .* proportions, 2), dCdP);%代价函数对权重wi的更新
  val dCdW: DenseVector[Double] = proportions(idx, ::).t :* (sum(dCdP :* proportions(idx, ::).t) - dCdP)
  (dCdW, idx)
 }



 private  def ComputedCdP(proportions: DenseVector[Double],
                          numer: DenseVector[Double],
                          tmp: DenseVector[Double]): Double = {
  sum(proportions :* numer :* tmp)
 }

 //PQ = Q - P
 def GetPQ(P: (Int, Iterable[(Int,Double)]), Q: DenseVector[Double]): DenseVector[Double] = {
  val PQ = Q.copy
  P match {
   case (_, itr) =>
    itr.foreach {
     case (j, value) => {
      PQ.update(j, Q(j) - value)
     }
    }
  }
  PQ
 }

 def GettempPQ(QZ: DenseVector[Double], PQ:  DenseVector[Double]): DenseVector[Double] = {
  (1.0 :/ QZ) :* PQ
 }

 def GetQ(QZ: DenseVector[Double], Z: Double): DenseVector[Double] = {
  QZ :/ Z
 }

 def GetL(P: (Int, Iterable[(Int, Double)]),n : Int): DenseVector[Double] = {
  // 每一行所有的元素求和, 放在对角线上然后减去这一行上所有元素
  val value = P._2.aggregate(0.0)(
   seqop = (c,v) => c + v._2,
   combop = _ + _
  )
  val Lvector = DenseVector.fill(n){realmin}
  val index = P._1
  Lvector.update(index, value)
  P._2.foreach{
   case (j, v1) => {
    val old = Lvector(j)
    Lvector.update(j, old - v1)
   }
  }
  Lvector
 }


 def ZoomP(P: (Int,Iterable[(Int,Double)]), zoomValue: Double, exaggeration: Boolean): (Int,Iterable[(Int,Double)]) = {
  if (exaggeration)
   (P._1,P._2.map(arr => (arr._1, arr._2 * zoomValue)))
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

 def Compute_loss_end(proportions: DenseMatrix[Double], L: DenseVector[Double], laplace: Double, index: Int): Double = {
  laplace * sum(proportions.t * L * proportions(index,::))
 }

}
