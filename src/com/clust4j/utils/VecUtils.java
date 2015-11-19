package com.clust4j.utils;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

public class VecUtils {
	public final static int MIN_ACCEPTABLE_VEC_LEN = 1;
	public final static boolean DEF_SUBTRACT_ONE_VAR = true;
	
	final static public void checkDims(final double[] a) {
		if(a.length < MIN_ACCEPTABLE_VEC_LEN) throw new IllegalArgumentException("illegal vector length");
	}
	
	final static public void checkDims(final double[] a, final double[] b) {
		if(a.length != b.length) throw new DimensionMismatchException(a.length, b.length);
		checkDims(a); // Only need to do one, knowing they are same length
	}
	
	
	
	
	public static double[] abs(final double[] a) {
		final double[] b= new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.abs(a[i]);
		return b;
	}
	
	public static double[] add(final double[] a, final double[] b) {
		checkDims(a, b);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] + b[i];
		
		return ab;
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
	
	public static int[] copy(final int[] i) {
		final int[] copy = new int[i.length];
		System.arraycopy(i, 0, copy, 0, i.length);
		return copy;
	}
	
	public static double[] copy(final double[] d) {
		final double[] copy = new double[d.length];
		System.arraycopy(d, 0, copy, 0, d.length);
		return copy;
	}
	
	public static double cosSim(final double[] a, final double[] b) {
		checkDims(a, b);
		
		// Calculate all in one to avoid O(3N)
		double innerProdSum = 0;
		double normAsum = 0;
		double normBsum = 0;
		
		for(int i = 0; i < a.length; i++) {
			innerProdSum += a[i] * b[i];
			normAsum += a[i] * a[i];
			normBsum += b[i] * b[i];
		}
		
		return innerProdSum / (FastMath.sqrt(normAsum) * FastMath.sqrt(normBsum));
	}
	
	public static double[] divide(final double[] numer, final double[] by) {
		checkDims(numer, by);
		
		final double[] ab = new double[numer.length];
		for(int i = 0; i < numer.length; i++)
			ab[i] = numer[i] / by[i];
		
		return ab;
	}
	
	public static boolean equalsExactly(final double[] a, final double[] b) {
		checkDims(a, b);
		
		for(int i = 0; i < a.length; i++)
			if(a[i] != b[i])
				return false;
		
		return true;
	}
	
	public static boolean equalsWithTolerance(final double[] a, final double[] b) {
		return equalsWithTolerance(a, b, Precision.EPSILON);
	}
	
	public static boolean equalsWithTolerance(final double[] a, final double[] b, final double eps) {
		checkDims(a, b);
		
		for(int i = 0; i < a.length; i++)
			if( !Precision.equals(a[i], b[i], eps) )
				return false;
		
		return true;
	}
	
	public static double innerProduct(final double[] a, final double[] b) {
		checkDims(a, b);
		
		double sum = 0d;
		for(int i = 0; i < a.length; i++)
			sum += a[i] * b[i];
		
		return sum;
	}
	
	public static boolean isOrthogonalTo(final double[] a, final double[] b) {
		checkDims(a, b);
		return Precision.equals(innerProduct(a, b), 0, Precision.EPSILON);
	}
	
	public static double l1Norm(final double[] a) {
		return sum(abs(a));
	}
	
	public static double l2Norm(final double[] a) {
		return FastMath.sqrt(innerProduct(a, a));
	}
	
	public static double[] log(final double[] a) {
		final double[] b = new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.log(a[i]);
		return b;
	}
	
	public static double lpNorm(final double[] a, final double p) {
		if(p == 1) return l1Norm(a);
		if(p == 2) return l2Norm(a);
		
		double power = 1.0 / p;
		return FastMath.pow(sum(abs(a)), power);
	}
	
	public static double magnitude(final double[] a) {
		return l2Norm(a);
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
	
	public static double[] multiply(final double[] a, final double[] b) {
		checkDims(a, b);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] * b[i];
		
		return ab;
	}
	
	/**
	 * Scalar divides a vector by its magnitude
	 * @param a
	 * @return
	 */
	public static double[] normalize(double[] a) {
		return scalarDivide(a, magnitude(a));
	}
	
	public static double[][] outerProduct(final double[] a, final double[] b) {
		// Can be different lengths...
		checkDims(a);
		checkDims(b);
		
		final double[][] ab = new double[a.length][];
		for(int i = 0; i < a.length; i++) {
			final double[] row = new double[b.length];
			for(int j = 0; j < b.length; j++)
				row[j] = a[i] * b[j];
			
			ab[i] = row;
		}
		
		return ab;
	}
	
	/**
	 * Returns a vector of the max parallel elements in each respective vector
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] pmax(final double[] a, final double[] b) {
		checkDims(a, b);
		
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++)
			out[i] = FastMath.max(a[i], b[i]);
		
		return out;
	}
	
	/**
	 * Returns a vector of the min parallel elements in each respective vector
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] pmin(final double[] a, final double[] b) {
		checkDims(a, b);
		
		final double[] out = new double[a.length];
		for(int i = 0; i < a.length; i++)
			out[i] = FastMath.min(a[i], b[i]);
		
		return out;
	}
	
	public static double[] pow(final double[] a, final double p) {
		final double[] b = new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.pow(a[i], p);
		return b;
	}
	
	public static double prod(final double[] a) {
		checkDims(a);
		double prod = 1;
		for(double d: a)
			prod *= d;
		return prod;
	}
	
	public static double[] scalarAdd(final double[] a, final double b) {
		checkDims(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] + b;
		
		return ab;
	}
	
	public static double[] scalarDivide(final double[] a, final double b) {
		checkDims(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] / b;
		
		return ab;
	}
	
	public static double[] scalarMultiply(final double[] a, final double b) {
		checkDims(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] * b;
		
		return ab;
	}
	
	public static double[] scalarSubtract(final double[] a, final double b) {
		checkDims(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] - b;
		
		return ab;
	}
	
	public static double[] sqrt(final double[] a) {
		final double[] b = new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.sqrt(a[i]);
		return b;
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
	
	public static double[] subtract(final double[] from, final double[] subtractor) {
		checkDims(from, subtractor);
		
		final double[] ab = new double[from.length];
		for(int i = 0; i < from.length; i++)
			ab[i] = from[i] - subtractor[i];
		
		return ab;
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
