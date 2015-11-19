package com.clust4j.kernel;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.SimilarityMetric;
import com.clust4j.utils.VecUtils;

public abstract class Kernel implements SimilarityMetric {
	
	public Kernel() {}
	
	
	/* public final double similarity(final double[] a) {
		return distance(a, a);
	} */
	
	@Override
	final public double getDistance(final double[] a, final double[] b) {
		return -getSimilarity(a, b);
	}
	
	
	protected static double toHilbertPSpace(final double[] a, final double[] b) {
		return 2*VecUtils.innerProduct(a, b) - VecUtils.innerProduct(a,a) - VecUtils.innerProduct(b,b);
	}
	
	
	public double[][] kernelDistanceMatrix(final AbstractRealMatrix a) {
		return kernelDistanceMatrix(a.getData());
	}
	
	public double[][] kernelDistanceMatrix(final double[][] data) {
		return ClustUtils.distanceMatrix(data, this);
	}
	
	
	public double[][] kernelSimilarityMatrix(final AbstractRealMatrix a) {
		return kernelSimilarityMatrix(a.getData());
	}
	
	public double[][] kernelSimilarityMatrix(final double[][] data) {
		return ClustUtils.similarityMatrix(data, this);
	}
	
	
	@Override
	public String toString() {
		return getName();
	}
}
