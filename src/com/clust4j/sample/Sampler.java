package com.clust4j.sample;

import java.util.Random;

public interface Sampler {
	public double[][] sample(final double[][] data, final int n);
	public double[][] sample(final double[][] data, final int n, final Random seed);
}
