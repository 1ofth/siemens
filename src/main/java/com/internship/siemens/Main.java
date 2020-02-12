package com.internship.siemens;

import java.util.Arrays;

public class Main {
    private static final byte EX_USAGE = 64;

    public static void main(final String[] args) {
        if (args.length < 4) {
            System.err.println("Not enough arguments! Expected 4:");
            System.err.println("    path to file(s)");
            System.err.println("    learning rate");
            System.err.println("    accuracy");
            System.err.println("    max iterations");
            System.exit(EX_USAGE);
        }

        double[] weights =
                DistributedGradientDescent.train(args[0],
                        Double.parseDouble(args[1]), Double.parseDouble(args[2]),
                        Integer.parseInt(args[3]));

        System.out.println("Trained weights:");
        System.out.println(Arrays.toString(weights));
    }
}
