package com.clust4j.utils;

public class CosineSimilarity implements GeometricallySeparable {
	public CosineSimilarity() {}

	@Override
	public double distance(double[] a, double[] b) {
		return VecUtils.cosSim(a, b);
	}

	@Override
	public String getName() {
		return "Cosine Similarity";
	}
}
