package com.clust4j.sample;

import java.util.Random;

import com.clust4j.NamedEntity;

public interface Sampler extends NamedEntity {
	public double[][] sample(final double[][] data, final int n);
	public double[][] sample(final double[][] data, final int n, final Random seed);
}
