# movie_recommendation_ALS

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit 

HDFS_RAW_DATA="hdfs://localhost:9001/raw/u.data"

# Load up moview ID -> movie name dictionary 
def loadMovieNames():
    movie_names = {}
    with open("ml-100k/u.item", encoding='ISO-8859-1') as file:
        for line in file:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]

    return movie_names

# Convert u.data lines into (userId, movieID, rating) rows
def parseInput(line):
    fields = line.value.split() # value? 
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))

if __name__ == "__main__":
    # Create a Sparksession the config bit is only for windows!
    spark = SparkSession.builder.appName("Movie Recommend").getOrCreate()

    movie_names = loadMovieNames()

    lines = spark.read.text(HDFS_RAW_DATA).rdd

    ratingsRDD = lines.map(parseInput)

    ratings = spark.createDataFrame(ratingsRDD).cache()

    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
    model = als.fit(ratings)

    print("\nRatings for user ID 0: ")
    user_ratings = ratings.filter("userID = ")
    for rating in user_ratings.collect():
        print(movie_names[rating['movieID']], rating['rating'])

    print("\nTop 20 recommendations: ")
    rating_counts = ratings.groupBy("movieID").count().filter("count > 100")
    popular_movies = rating_counts.select("movieID").withColumn('userID', lit(0))

    recommendations = model.transform(popular_movies)

    top_recommendations = recommendations.sort(recommendations.prediction.desc()).take(20)

    for recommendation in top_recommendations:
        print(movie_names[recommendation['movieID']], recommendation['prediction'])

    spark.stop()


