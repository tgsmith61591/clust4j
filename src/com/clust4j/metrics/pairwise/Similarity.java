package com.clust4j.metrics.pairwise;

import com.clust4j.utils.VecUtils;

public enum Similarity implements SimilarityMetric {
	COSINE {
		@Override public double getDistance(final double[] a, final double[] b) {
			return -getSimilarity(a, b);
		}
		
		@Override public double getSimilarity(final double[] a, final double[] b) {
			return VecUtils.cosSim(a, b);
		}
		
		@Override public String getName() {
			return "Cosine Similarity";
		}
	}
}
