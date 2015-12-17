package com.clust4j.sample;

import java.util.Random;

import com.clust4j.utils.Named;

public interface Sampler extends Named {
	public double[][] sample(final double[][] data, final int n);
	public double[][] sample(final double[][] data, final int n, final Random seed);
}
