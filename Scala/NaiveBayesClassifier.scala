object NaiveBayesClassifier {
  def train(X: Array[Array[Int]], y: Array[String]): NaiveBayesModel = {
    val classes = y.distinct
    val classProbabilities = classes.map(cls => cls -> y.count(_ == cls).toDouble / y.length).toMap
    val featureProbabilities = classes.map(cls => cls -> Array.tabulate(X(0).length)(j => X.zip(y).count { case (features, label) => features(j) == 1 && label == cls }.toDouble / y.count(_ == cls))).toMap
    NaiveBayesModel(classes, classProbabilities, featureProbabilities)
  }

  def predict(model: NaiveBayesModel, X: Array[Array[Int]]): Array[String] = X.map(features => model.classes.maxBy(cls => model.classProbabilities(cls) * features.indices.map(j => model.featureProbabilities(cls)(j)).product))
}

case class NaiveBayesModel(classes: Array[String], classProbabilities: Map[String, Double], featureProbabilities: Map[String, Array[Double]])

val X = Array(Array(1, 0), Array(1, 1), Array(0, 1), Array(0, 0))
val y = Array("Spam", "Spam", "Ham", "Ham")
val model = NaiveBayesClassifier.train(X, y)
println("Predictions: " + NaiveBayesClassifier.predict(model, X).mkString(", "))
