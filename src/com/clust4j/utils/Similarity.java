package com.clust4j.utils;

public enum Similarity implements GeometricallySeparable, SimilarityMetric {
	COSINE {
		@Override public double getSeparability(final double[] a, final double[] b) {
			return VecUtils.cosSim(a, b);
		}
		
		@Override public String getName() {
			return "Cosine Similarity";
		}
	}
}
