package com.internship.siemens;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.util.CollectionAccumulator;
import org.apache.spark.util.DoubleAccumulator;

public class GradientDescent {
    private static final byte EX_USAGE = 64;
    private static final double ACCURACY = 0.0001;

    private GradientDescent() {
    }

    static double hypothesis(final double[] xy, final double[] weights) {
        double hypothesis = weights[0];
        for (int i = 0; i < xy.length - 1; i++) {
            hypothesis += weights[i + 1] * xy[i];
        }
        return hypothesis;
    }

    static JavaRDD<double[]> divideRDD(final JavaRDD<String> input) {
        //@TODO what is preferable in our case? map or mapPartitions? JavaRDD or JavaPairRDD???
        return input.map(e ->
                Arrays.stream(e.split("\\p{javaSpaceChar}+"))
                        .mapToDouble(Double::valueOf)
                        .toArray());
    }

    static void updateWeights(final double[] weights,
                              final CollectionAccumulator<double[]> thetaAccumulator,
                              final double learningRate,
                              final long dataSetSize) {
        List<double[]> localThetaValues = thetaAccumulator.value();
        double[] thetaValue = new double[weights.length];

        for (double[] currThetaValues : localThetaValues) {
            assert thetaValue.length == currThetaValues.length;
            for (int i = 0; i < currThetaValues.length; i++) {
                thetaValue[i] += currThetaValues[i];
            }
        }

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate / dataSetSize * thetaValue[i];
        }
    }

    static void handlePartition(final Iterator<double[]> iterator,
                                final DoubleAccumulator errorAccumulator,
                                final CollectionAccumulator<double[]> thetaAccumulator,
                                final double[] weights) {
        double[] localGradient = new double[weights.length];
        double[] localError = new double[1];
        Arrays.fill(localGradient, 0);
        localError[0] = 0;
        iterator.forEachRemaining(item -> { // foreach sample in DS, item - array of x's and y
            double delta = hypothesis(item, weights) - item[item.length - 1];
            for (int i = 0; i < localGradient.length; i++) {
                localGradient[i] += delta * item[i];
            }
            localError[0] += delta * delta;

        });
        System.out.println("Local error: " + localError[0]);
        // send error and grad to master
        thetaAccumulator.add(localGradient);
        errorAccumulator.add(localError[0]);
    }

    static void gradientDescent(final double[] weights,
                                final JavaSparkContext jsc,
                                final JavaRDD<double[]> dataset,
                                final double learningRate,
                                final int maxIterations) {
        Arrays.fill(weights, 1);
        double errorPrevious = Double.POSITIVE_INFINITY;
        double errorCurrent = 0;
        int iteration = 0;
        long dataSetSize = dataset.count();
        System.out.println(dataset.getNumPartitions());
        DoubleAccumulator errorAccumulator = jsc.sc().doubleAccumulator();
        CollectionAccumulator<double[]> thetaAccumulator = jsc.sc().collectionAccumulator();

        while (Math.abs(errorPrevious - errorCurrent) > ACCURACY && iteration++ < maxIterations) {

            dataset.foreachPartition(iterator ->
                    handlePartition(iterator, errorAccumulator, thetaAccumulator, weights));

            errorPrevious = errorCurrent;
            errorCurrent = errorAccumulator.value() / dataSetSize;

            System.out.println("Current error: " + errorCurrent);

            updateWeights(weights, thetaAccumulator, learningRate, dataSetSize);

            errorAccumulator.reset();
            thetaAccumulator.reset();
        }
    }

    /**
     * @return trained weights
     */
    public static double[] distributedGradientDescent(final String path,
                                               final int coresNumber,
                                               final double learningRate,
                                               final int maxIterations) {
        // when running using spark-submit
        SparkConf conf = new SparkConf()
                .setAppName("DistributedGradientDescent")
                .setMaster(String.format("local[%1$d]", coresNumber));
        JavaSparkContext jsc = new JavaSparkContext(conf);
        JavaRDD<String> incoming = jsc.textFile(path);
        JavaRDD<double[]> dataset = divideRDD(incoming);
        double[] weights = new double[dataset.first().length];

        gradientDescent(weights, jsc, dataset, learningRate, maxIterations);

        jsc.stop();
        return weights;

    }

    public static void main(final String[] args) {
        //@TODO error handling!!!
        if (args.length < 4) {
            System.err.println("Not enough arguments! Expected 4.");
            System.exit(EX_USAGE);
        }
        double[] weights =
                distributedGradientDescent(args[0], Integer.parseInt(args[1]),
                        Double.parseDouble(args[2]), Integer.parseInt(args[3]));
        System.out.println(Arrays.toString(weights));
    }
}