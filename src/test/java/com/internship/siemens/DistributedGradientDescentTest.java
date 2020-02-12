package com.internship.siemens;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

public class DistributedGradientDescentTest {

    private JavaSparkContext jsc;

    @Before
    public void setUp() {
        SparkConf conf = new SparkConf()
                .setAppName("DistributedGradientDescent")
                .setMaster("local[*]");
        this.jsc = new JavaSparkContext(conf);
    }

    @After
    public void tearDown() {
        jsc.stop();
    }

    @Test
    public void predictionTest() {
        double[] parameters = new double[]{2, 3, 4};
        double[] weights = new double[]{4, 3, 2};
        double expected = 2 * 4 + 3 * 3 + 4 * 2;
        assertEquals(expected, DistributedGradientDescent.predict(parameters, weights), 1E-4);
    }

    @Test
    public void extractParametersFromRDDTest() {
        List<String> str = new ArrayList<>();
        str.add("2  3  4");
        JavaRDD<String> ds = jsc.parallelize(str);

        JavaPairRDD<double[], Double> actual = DistributedGradientDescent.extractParametersFromRDD(ds);

        List<Tuple2<double[], Double>> result = actual.collect();

        assertEquals(1, result.size());
        assertArrayEquals(new double[]{1, 2, 3}, result.get(0)._1, 1E-5);
        assertEquals(4d, result.get(0)._2, 1E-5);
    }

}
