import breeze.linalg._
import breeze.numerics.{exp, pow, sqrt}

val fuck = DenseMatrix.fill(3,4){1.0}
fuck(1,::) := DenseVector(Array(2.0,2.0,2.0,2.0)).t
fuck(2,::) := DenseVector(Array(3.0,3.0,3.0,3.0)).t
val fuck1 = sum(fuck.apply(::,*))
val thu = DenseVector(Array(1.0,2.0,3.0,4.0))
fuck.apply(*,::) :/ sum(fuck.apply(::,*)).t
