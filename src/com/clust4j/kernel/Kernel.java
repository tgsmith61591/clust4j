package com.clust4j.kernel;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.VecUtils;

public abstract class Kernel implements SimilarityMetric {
	private static final long serialVersionUID = -630865804908845073L;


	public Kernel() {}
	
	
	
	@Override
	public double getDistance(final double[] a, final double[] b) {
		return -getSimilarity(a, b);
	}
	
	@Override
	public double getPartialDistance(final double[] a, final double[] b) {
		return -getPartialSimilarity(a, b);
	}
	
	@Override
	public double getPartialSimilarity(final double[] a, final double[] b) {
		return getSimilarity(a, b);
	}
	
	@Override
	public double distanceToPartialDistance(double d) {
		return -similarityToPartialSimilarity(-d);
	}
	
	@Override
	public double similarityToPartialSimilarity(double d) {
		return d;
	}
	
	@Override
	public double partialDistanceToDistance(double d) {
		return -partialSimilarityToSimilarity(-d);
	}
	
	@Override
	public double partialSimilarityToSimilarity(double d) {
		return d;
	}
	
	
	
	protected static double toHilbertPSpace(final double[] a, final double[] b) {
		// Originally: 2*VecUtils.innerProductForceSerial(a, b) - VecUtils.innerProduct(a,a) - VecUtils.innerProduct(b,b);
		// This costs 3N!!
		VecUtils.checkDims(a,b);
		double ipab = 0, ipaa = 0, ipbb = 0;
		int n = a.length;
		
		// This only costs 1N but is uglier...
		for(int i = 0; i < n; i++) {
			ipab += a[i] * b[i];
			ipaa += a[i] * a[i];
			ipbb += b[i] * b[i];
		}
		
		//return 2*VecUtils.innerProductForceSerial(a, b) - VecUtils.innerProduct(a,a) - VecUtils.innerProduct(b,b);
		return 2*ipab - ipaa - ipbb;
	}
	
	
	/**
	 * Returns an upper triangular distance matrix computed using this kernel
	 * @param a
	 * @return UT dist matrix
	 */
	public double[][] kernelDistanceMatrixUT(final AbstractRealMatrix a) {
		return kernelDistanceMatrixUT(a.getData());
	}
	
	/**
	 * Returns an upper triangular distance matrix computed using this kernel
	 * @param data
	 * @return UT dist matrix
	 */
	public double[][] kernelDistanceMatrixUT(final double[][] data) {
		return ClustUtils.distanceUpperTriangMatrix(data, this);
	}
	
	/**
	 * Returns a full distance matrix computed using this kernel
	 * @param a
	 * @return full dist matrix
	 */
	public double[][] kernelDistanceMatrixFull(final AbstractRealMatrix a) {
		return kernelDistanceMatrixFull(a.getData());
	}
	
	/**
	 * Returns a full distance matrix computed using this kernel
	 * @param data
	 * @return full dist matrix
	 */
	public double[][] kernelDistanceMatrixFull(final double[][] data) {
		return ClustUtils.distanceFullMatrix(data, this);
	}
	
	/**
	 * Returns an upper triangular similarity matrix computed using this kernel
	 * @param a
	 * @return UT similarity matrix
	 */
	public double[][] kernelSimilarityMatrixUT(final AbstractRealMatrix a) {
		return kernelSimilarityMatrixUT(a.getData());
	}
	
	/**
	 * Returns an upper triangular similarity matrix computed using this kernel
	 * @param data
	 * @return UT similarity matrix
	 */
	public double[][] kernelSimilarityMatrixUT(final double[][] data) {
		return ClustUtils.similarityUpperTriangMatrix(data, this);
	}
	
	/**
	 * Returns a full similarity matrix computed using this kernel
	 * @param a
	 * @return full similarity matrix
	 */
	public double[][] kernelSimilarityMatrixFull(final AbstractRealMatrix a) {
		return kernelSimilarityMatrixFull(a.getData());
	}
	
	/**
	 * Returns a full similarity matrix computed using this kernel
	 * @param data
	 * @return full similarity matrix
	 */
	public double[][] kernelSimilarityMatrixFull(final double[][] data) {
		return ClustUtils.similarityFullMatrix(data, this);
	}
	
	/**
	 * Returns the name of the kernel
	 */
	@Override
	public String toString() {
		return getName();
	}
}
