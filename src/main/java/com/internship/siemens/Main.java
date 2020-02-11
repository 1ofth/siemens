package com.internship.siemens;

import java.util.Arrays;

import static com.internship.siemens.GradientDescent.distributedGradientDescent;

public class Main {
    private static final byte EX_USAGE = 64;

    public static void main(final String[] args) {
        //@TODO error handling!!!
        if (args.length < 4) {
            System.err.println("Not enough arguments! Expected 4.");
            System.exit(EX_USAGE);
        }
        double[] weights =
                distributedGradientDescent(args[0], Double.parseDouble(args[1]),
                        Integer.parseInt(args[2]), Integer.parseInt(args[3]));

        System.out.println("Trained weights:");
        System.out.println(Arrays.toString(weights));
    }
}
