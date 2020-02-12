package com.internship.siemens;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.util.CollectionAccumulator;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;

public class DistributedGradientDescent {

    /**
     * Calculates function value by given weights and parameters
     *
     * @param params  parameters array
     * @param weights weights array
     * @return calculated value
     */
    public static double predict(final double[] weights, final double[] params) {
        double hypothesis = 0;
        for (int i = 0; i < params.length; i++) {
            hypothesis += weights[i] * params[i];
        }
        return hypothesis;
    }

    /**
     * Transforms JavaRDD<String> to JavaPairRDD<double[], Double>, where double[] it is an array containing
     * parameters (with leading 1.0) and Double it is a function of parameters.
     *
     * @param input JavaRDD<String> object to parse
     * @return a JavaPairRDD<double[], Double> object containing parameters array and result value
     */
    static JavaPairRDD<double[], Double> extractParametersFromRDD(final JavaRDD<String> input) {
        return input.mapPartitionsToPair(itr -> {
            List<Tuple2<double[], Double>> list = new ArrayList<>();
            itr.forEachRemaining(line -> {
                String[] parameters = "1 ".concat(line.trim()).split("\\p{javaSpaceChar}+");

                list.add(new Tuple2<>(
                        Arrays.stream(parameters, 0, parameters.length - 1)
                                .mapToDouble(Double::valueOf)
                                .toArray(), Double.valueOf(parameters[parameters.length - 1])));

            });
            return list.iterator();
        });
    }

    /**
     * @param path          to text file containing dataset
     * @param learningRate  learning rate, alpha constant in theoretical formulas
     * @param accuracy      when error changes less then accuracy, computations stop
     * @param maxIterations max number of method iterations
     * @return trained weights
     */
    public static double[] train(final String path,
                                 final double learningRate,
                                 final double accuracy,
                                 final int maxIterations) {
        SparkConf config = new SparkConf();
        config.setAppName("DistributedGradientDescent");
        JavaSparkContext jsc =
                new JavaSparkContext(config);

        JavaRDD<String> incoming = jsc.textFile(path);
        JavaPairRDD<double[], Double> dataset = extractParametersFromRDD(incoming);
        double[] weights = new double[dataset.first()._1.length];
        Arrays.fill(weights, 1);
        gradientDescentRunner(weights, jsc, dataset, learningRate, accuracy, maxIterations);
        jsc.stop();
        return weights;
    }

    private static void gradientDescentRunner(final double[] weights,
                                              final JavaSparkContext jsc,
                                              final JavaPairRDD<double[], Double> dataset,
                                              final double learningRate,
                                              final double accuracy,
                                              final int maxIterations) {
        double errorPrevious = Double.POSITIVE_INFINITY;
        double errorCurrent = 0;
        int iteration = 0;
        long dataSetSize = dataset.count();
        DoubleAccumulator errorAccumulator = jsc.sc().doubleAccumulator();
        CollectionAccumulator<double[]> gradientAccumulator = jsc.sc().collectionAccumulator();

        while (Math.abs(errorPrevious - errorCurrent) > accuracy && iteration++ < maxIterations) {

            stepGradientDescent(dataset, errorAccumulator, gradientAccumulator, weights);
            updateWeights(weights, gradientAccumulator.value(), learningRate, dataSetSize);

            errorPrevious = errorCurrent;
            errorCurrent = errorAccumulator.value() / dataSetSize;

            errorAccumulator.reset();
            gradientAccumulator.reset();
        }
    }

    private static void stepGradientDescent(final JavaPairRDD<double[], Double> dataset,
                                            final DoubleAccumulator errorAccumulator,
                                            final CollectionAccumulator<double[]> gradientAccumulator,
                                            final double[] weights) {
        dataset.foreachPartition(iterator -> {
            double[] localGradient = new double[weights.length];
            double[] localError = new double[1];
            Arrays.fill(localGradient, 0);
            localError[0] = 0;
            iterator.forEachRemaining(item -> {
                double delta = predict(item._1, weights) - item._2;
                for (int i = 0; i < localGradient.length; i++) {
                    localGradient[i] += delta * item._1[i];
                }
                localError[0] += delta * delta;

            });
            // send local error and local gradient to master
            gradientAccumulator.add(localGradient);
            errorAccumulator.add(localError[0]);
        });
    }

    private static void updateWeights(final double[] weights,
                                      final List<double[]> localGradients,
                                      final double learningRate,
                                      final long dataSetSize) {
        double[] deltaGradient = new double[weights.length];

        for (double[] currentLocalGradient : localGradients) {
            for (int i = 0; i < currentLocalGradient.length; i++) {
                deltaGradient[i] += currentLocalGradient[i];
            }
        }

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate / dataSetSize * deltaGradient[i];
        }
    }
}
