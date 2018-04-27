import breeze._
import breeze.linalg.{*, DenseMatrix, DenseVector, sum, tile}

import scala.collection.mutable.ArrayBuffer

// weight 检查 -> 公式进行修改
val hello = DenseMatrix((1.0,1.0,1.0,1.0,1.0),(2.0,2.0,2.0,2.0,2.0),(3.0,3.0,3.0,3.0,3.0))
val st = sum(hello.apply(*,::))
val st1 = hello(::,*)
val fuck = hello(::,*) :/ sum(hello(*,::))
/**
  * 0.2  0.2  0.2  0.2  0.2
    0.2  0.2  0.2  0.2  0.2
    0.2  0.2  0.2  0.2  0.2  => fuck 最后的结果
  * */

//numerator 检查
//检查正确

// QQ QZ 检查
// 检查正确

// dCdD 检查
  //-> 检查 PQ 检查正确
// dCdD 检查正确

// dCdY 检查
//sum((tile(dCdD_m, 1, no_dims) :* (tile(Y_m(index, ::), 1, n) - Y_m)).apply(::, *)).t
//Y_m(index,::) -> 检查正确
hello + tile(hello(1,::),1,3)
// dCdY 检查正确

// dCdW 检查
    //-> dCdP 检查
        //-> temPQ 检查
// dCdW 检查


// 检查拉普拉斯矩阵
// 拉普拉斯检查正确

val futureConcept = new ArrayBuffer[String]()
futureConcept.append("hello")
futureConcept.append("bitch")
futureConcept.toArray
println(futureConcept.length)