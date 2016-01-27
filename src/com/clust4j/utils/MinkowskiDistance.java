package com.clust4j.utils;

import org.apache.commons.math3.util.FastMath;

public class MinkowskiDistance implements DistanceMetric {
	private static final long serialVersionUID = 6206826797866732365L;
	final private double p;
	
	public MinkowskiDistance(final double p) {
		if(p < 1)
			throw new IllegalArgumentException("p cannot be less than 1");
		this.p = p;
	}

	@Override
	public double getDistance(double[] a, double[] b) {
		VecUtils.checkDims(a,b);
		
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			// Don't use math.abs -- too expensive
			double diff = a[i] - b[i];
			sum += FastMath.pow(FastMath.abs(diff), p);
		}
		
		return FastMath.pow(sum, 1d/p);
	}
	
	@Override
	public String getName() {
		return "Minkowski";
	}
}
