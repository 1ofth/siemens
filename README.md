# Internship task for Siemens

Implementing a distributed version of gradient descent in java using Apache Spark.

### Setup

```
mvn package
```
### Usage


```
$SPARK_HOME/bin/spark-submit --master local[*] target/siemens-1.0-SNAPSHOT.jar example_dataset.txt 0.00001 0.00001 1000
```