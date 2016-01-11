package com.clust4j.kernel;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.SimilarityMetric;
import com.clust4j.utils.VecUtils;

public abstract class Kernel implements SimilarityMetric {
	private static final long serialVersionUID = -630865804908845073L;


	public Kernel() {}
	
	
	/* public final double similarity(final double[] a) {
		return distance(a, a);
	} */
	
	@Override
	final public double getDistance(final double[] a, final double[] b) {
		return -getSimilarity(a, b);
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
	
	
	public double[][] kernelDistanceMatrix(final AbstractRealMatrix a) {
		return kernelDistanceMatrix(a.getData());
	}
	
	public double[][] kernelDistanceMatrix(final double[][] data) {
		return ClustUtils.distanceUpperTriangMatrix(data, this);
	}
	
	
	public double[][] kernelSimilarityMatrix(final AbstractRealMatrix a) {
		return kernelSimilarityMatrix(a.getData());
	}
	
	public double[][] kernelSimilarityMatrix(final double[][] data) {
		return ClustUtils.similarityUpperTriangMatrix(data, this);
	}
	
	
	@Override
	public String toString() {
		return getName();
	}
}
