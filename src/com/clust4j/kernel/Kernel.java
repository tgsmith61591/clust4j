package com.clust4j.kernel;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.SimilarityMetric;
import com.clust4j.utils.VecUtils;

public abstract class Kernel implements GeometricallySeparable, SimilarityMetric {
	
	public Kernel() {}
	
	
	/* public final double similarity(final double[] a) {
		return distance(a, a);
	} */
	
	
	protected static double toHilbertPSpace(final double[] a, final double[] b) {
		return 2*VecUtils.innerProduct(a, b) - VecUtils.innerProduct(a,a) - VecUtils.innerProduct(b,b);
	}
	
	public double[][] kernelMatrix(final AbstractRealMatrix a) {
		return kernelMatrix(a.getData());
	}
	
	public double[][] kernelMatrix(final double[][] data) {
		return ClustUtils.distanceMatrix(data, this);
	}
	
	@Override
	public String toString() {
		return getName();
	}
}
