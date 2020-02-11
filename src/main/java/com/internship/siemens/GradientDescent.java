package com.internship.siemens;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.util.CollectionAccumulator;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;

public class GradientDescent {
    private static final double ACCURACY = 1E-10;

    private GradientDescent() {
    }

    static double predict(final double[] params, final double[] weights) {
        double hypothesis = 0;
        for (int i = 0; i < params.length; i++) {
            hypothesis += weights[i] * params[i];
        }
        return hypothesis;
    }

    /**
     * Transforms JavaRDD<String> to JavaPairRDD<double[], Double>, where double[] is array containing
     * parameters (with leading 1.0) and Double is function of parameters.
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

    static void updateWeights(final double[] weights,
                              final CollectionAccumulator<double[]> thetaAccumulator,
                              final double learningRate,
                              final long dataSetSize) {
        List<double[]> localThetaValues = thetaAccumulator.value();
        double[] thetaValue = new double[weights.length];

        for (double[] currThetaValues : localThetaValues) {
            for (int i = 0; i < currThetaValues.length; i++) {
                thetaValue[i] += currThetaValues[i];
            }
        }

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate / dataSetSize * thetaValue[i];
        }
    }

    private static void stepGradientDescent(final JavaPairRDD<double[], Double> dataset,
                                            final DoubleAccumulator errorAccumulator,
                                            final CollectionAccumulator<double[]> thetaAccumulator,
                                            final double[] weights) {
        dataset.foreachPartition(iterator -> {
            double[] localGradient = new double[weights.length];
            double[] localError = new double[1];
            Arrays.fill(localGradient, 0);
            localError[0] = 0;
            iterator.forEachRemaining(item -> { // foreach sample in DS, item - array of x's and y
                double delta = predict(item._1, weights) - item._2;
                for (int i = 0; i < localGradient.length; i++) {
                    localGradient[i] += delta * item._1[i];
                }
                localError[0] += delta * delta;

            });
            // send error and grad to master
            thetaAccumulator.add(localGradient);
            errorAccumulator.add(localError[0]);
        });
    }

    private static void gradientDescentRunner(final double[] weights,
                                              final JavaSparkContext jsc,
                                              final JavaPairRDD<double[], Double> dataset,
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

            stepGradientDescent(dataset, errorAccumulator, thetaAccumulator, weights);
            updateWeights(weights, thetaAccumulator, learningRate, dataSetSize);

            errorPrevious = errorCurrent;
            errorCurrent = errorAccumulator.value() / dataSetSize;

            errorAccumulator.reset();
            thetaAccumulator.reset();
        }
    }

    /**
     *
     * @param path to text file containing dataset
     * @param learningRate learning rate, alpha constant in theoretical formulas
     * @param maxIterations max number of method iterations
     * @param coresNumber cores number to be used by spark
     * @return trained weights
     */
    public static double[] distributedGradientDescent(final String path,
                                                      final double learningRate,
                                                      final int maxIterations,
                                                      final int coresNumber) {
        // when running using spark-submit
        JavaSparkContext jsc =
                new JavaSparkContext(String.format("local[%1$d]", coresNumber),
                        "DistributedGradientDescent");
        JavaRDD<String> incoming = jsc.textFile(path);
        JavaPairRDD<double[], Double> dataset = extractParametersFromRDD(incoming);
        double[] weights = new double[dataset.first()._1.length];
        gradientDescentRunner(weights, jsc, dataset, learningRate, maxIterations);
        jsc.stop();
        return weights;
    }
}
