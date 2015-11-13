package com.clust4j.utils;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.FastMath;

public class VecUtils {
	public final static int MIN_ACCEPTABLE_VEC_LEN = 1;
	public final static boolean DEF_SUBTRACT_ONE_VAR = true;
	
	final static protected void checkDims(final double[] a, final double[] b) {
		if(a.length != b.length) throw new DimensionMismatchException(a.length, b.length);
		if(a.length < MIN_ACCEPTABLE_VEC_LEN) throw new IllegalArgumentException("illegal vector length");
	}
	
	final public static double[] center(final double[] a) {
		return center(a, mean(a));
	}
	
	final public static double[] center(final double[] a, final double mean) {
		final double[] copy = new double[a.length];
		System.arraycopy(a, 0, copy, 0, a.length);
		for(int i = 0; i < a.length; i++)
			copy[i] = a[i] - mean;
		return copy;
	}
	
	final public static double max(final double[] a) {
		double max = Double.MIN_VALUE;
		for(double d : a)
			if(d > max)
				max = d;
		return max;
	}
	
	final public static double mean(final double[] a) {
		return mean(a, sum(a));
	}
	
	final public static double mean(final double[] a, final double sum) {
		return sum / a.length;
	}
	
	final public static double min(final double[] a) {
		double min = Double.MAX_VALUE;
		for(double d : a)
			if(d < min)
				min = d;
		return min;
	}
	
	final public static double stdDev(final double[] a) {
		return stdDev(a, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double stdDev(final double[] a, final double mean) {
		return stdDev(a, mean, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double stdDev(final double[] a, final boolean n_minus_one) {
		return stdDev(a, mean(a), n_minus_one);
	}
	
	final public static double stdDev(final double[] a, final double mean, final boolean n_minus_one) {
		return FastMath.sqrt(var(a, mean, n_minus_one));
	}
	
	final public static double sum(final double[] a) {
		double sum = 0d;
		for(double d : a)
			sum += d;
		return sum;
	}
	
	final public static double var(final double[] a) {
		return var(a, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double var(final double[] a, final double mean) {
		return var(a, mean, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double var(final double[] a, final boolean n_minus_one) {
		return var(a, mean(a), n_minus_one);
	}
	
	final public static double var(final double[] a, final double mean, final boolean n_minus_one) {
		double sum = 0;
		for(double x : a) {
			double res = x - mean; // Want to avoid math.pow...
			sum += res * res;
		}
		return sum / (a.length - (n_minus_one ? 1 : 0));
	}
}
