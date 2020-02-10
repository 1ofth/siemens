package com.internship.siemens;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.util.CollectionAccumulator;
import org.apache.spark.util.DoubleAccumulator;
import scala.Tuple2;

public class GradientDescent {
    private static final byte EX_USAGE = 64;
    private static final double ACCURACY = 1E-10;

    private GradientDescent() {
    }

    static double hypothesis(final double[] params, final double[] weights) {
        double hypothesis = 0;
        for (int i = 0; i < params.length; i++) {
            hypothesis += weights[i] * params[i];
        }
        return hypothesis;
    }

    static JavaPairRDD<double[], Double> divideRDD(final JavaRDD<String> input) {
        return input.mapPartitionsToPair(itr -> {
            List<Tuple2<double[], Double>> list = new ArrayList<>();
            itr.forEachRemaining(line -> {
                String[] str = line.trim().split("\\p{javaSpaceChar}+");

                String[] parameters = new String[str.length];
                System.arraycopy(str, 0, parameters, 1, str.length - 1);
                parameters[0] = "1";

                list.add(new Tuple2<>(
                        Arrays.stream(parameters)
                                .mapToDouble(Double::valueOf)
                                .toArray(), Double.valueOf(str[str.length - 1])));

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

    static void stepGradientDescent(final JavaPairRDD<double[], Double> dataset,
                                    final DoubleAccumulator errorAccumulator,
                                    final CollectionAccumulator<double[]> thetaAccumulator,
                                    final double[] weights) {
        dataset.foreachPartition(iterator -> {
            double[] localGradient = new double[weights.length];
            double[] localError = new double[1];
            Arrays.fill(localGradient, 0);
            localError[0] = 0;
            iterator.forEachRemaining(item -> { // foreach sample in DS, item - array of x's and y
                double delta = hypothesis(item._1, weights) - item._2;
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
        JavaRDD<String> incoming = jsc.textFile(path, coresNumber);
        JavaPairRDD<double[], Double> dataset = divideRDD(incoming);
        double[] weights = new double[dataset.first()._1.length];

        gradientDescentRunner(weights, jsc, dataset, learningRate, maxIterations);

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
