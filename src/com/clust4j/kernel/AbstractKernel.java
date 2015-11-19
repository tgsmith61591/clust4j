package com.clust4j.kernel;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.VecUtils;

public abstract class AbstractKernel implements GeometricallySeparable {
	
	public AbstractKernel() {}
	
	
	/* public final double similarity(final double[] a) {
		return distance(a, a);
	} */
	
	
	protected static double getLpNorm(final double[] a, final double[] b, final double p) {
		return VecUtils.lpNorm(VecUtils.subtract(a, b), p);
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
