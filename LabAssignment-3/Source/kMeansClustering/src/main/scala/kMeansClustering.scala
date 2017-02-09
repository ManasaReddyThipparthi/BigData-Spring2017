import org.apache.log4j.{Logger, Level}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by Manasa on 09/02/2017.
  */
object kMeansClustering {

  def main(args: Array[String]): Unit = {

    System.setProperty("hadoop.home.dir","D:\\UMKC\\BigData\\LabTutorials\\Tutorial-2\\WinUtils");

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    val data = sc.textFile("data/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    parsedData.foreach(f=>println(f))
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    println("Clustering on training data: ")
    clusters.predict(parsedData).zip(parsedData).foreach(f=>println(f._2,f._1))
    clusters.save(sc, "data/KMeansModel")
    val sameModel = KMeansModel.load(sc, "data/KMeansModel")
  }
}
