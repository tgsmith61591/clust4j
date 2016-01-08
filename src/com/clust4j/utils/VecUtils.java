package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

import com.clust4j.GlobalState;
import com.clust4j.utils.parallel.map.DistributedAbs;
import com.clust4j.utils.parallel.map.DistributedAdd;
import com.clust4j.utils.parallel.map.DistributedMultiply;
import com.clust4j.utils.parallel.reduce.DistributedEqualityTest;
import com.clust4j.utils.parallel.reduce.DistributedInnerProduct;
import com.clust4j.utils.parallel.reduce.DistributedNaNCheck;
import com.clust4j.utils.parallel.reduce.DistributedNaNCount;
import com.clust4j.utils.parallel.reduce.DistributedProduct;
import com.clust4j.utils.parallel.reduce.DistributedSum;

import static com.clust4j.GlobalState.ParallelismConf.MAX_SERIAL_VECTOR_LEN;
import static com.clust4j.GlobalState.ParallelismConf.ALLOW_PARALLELISM;
import static com.clust4j.GlobalState.Mathematics.MAX;
import static com.clust4j.GlobalState.Mathematics.SIGNED_MIN;

public class VecUtils {
	public final static int MIN_ACCEPTABLE_VEC_LEN = 1;
	public final static boolean DEF_SUBTRACT_ONE_VAR = true;
	
	
	
	
	
	final static public void checkDims(final double[] a) {
		if(a.length < MIN_ACCEPTABLE_VEC_LEN) throw new IllegalArgumentException("illegal vector length:" + a.length);
	}
	
	final static public void checkDims(final double[] a, final double[] b) {
		if(a.length != b.length) throw new DimensionMismatchException(a.length, b.length);
		checkDims(a); // Only need to do one, knowing they are same length
	}
	
	final static public void checkDims(final int[] a) {
		if(a.length < MIN_ACCEPTABLE_VEC_LEN) throw new IllegalArgumentException("illegal vector length:" + a.length);
	}
	
	final static public void checkDims(final int[] a, final int[] b) {
		if(a.length != b.length) throw new DimensionMismatchException(a.length, b.length);
		checkDims(a); // Only need to do one, knowing they are same length
	}
	
	
	
	
	
	/**
	 * Calculate the absolute value of the values in the vector and return a copy.
	 * Depending on {@link GlobalState} parallelism settings, auto schedules parallel
	 * or serial job.
	 * @param a
	 * @return absolute value of the vector
	 */
	public static double[] abs(final double[] a) {
		checkDims(a);
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return absDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return absForceSerial(a);
	}
	
	
	/**
	 * Calculates the absolute value of the vector in a distributed fashion
	 * @param a
	 * @return absolute value of the vector
	 */
	public static double[] absDistributed(final double[] a) {
		return DistributedAbs.operate(a);
	}
	
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @return
	 */
	protected static double[] absForceSerial(final double[] a) {
		final double[] b= new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.abs(a[i]);
		return b;
	}
	
	
	/**
	 * Add two vectors and return a copy.
	 * Depending on {@link GlobalState} parallelism settings, auto schedules parallel
	 * or serial job.
	 * @param a
	 * @param b
	 * @return the result of adding two vectors
	 */
	public static double[] add(final double[] a, final double[] b) {
		checkDims(a, b);
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return addDistributed(a, b);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return addForceSerial(a, b);
	}
	
	
	/**
	 * Add two vectors and return a copy in a parallel fashion
	 * @param a
	 * @param b
	 * @return the result of adding two vectors
	 */
	public static double[] addDistributed(final double[] a, final double[] b) {
		return DistributedAdd.operate(a, b);
	}
	
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	protected static double[] addForceSerial(final double[] a, final double[] b) {
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] + b[i];
		
		return ab;
	}
	
	
	
	public static int[] arange(final int length) {
		return arange(0, length, 1);
	}
	
	public static int[] arange(final int start_inc, final int end_exc) {
		return arange(start_inc, end_exc, start_inc>end_exc?-1:1);
	}
	
	public static int[] arange(final int start_inc, final int end_exc, final int increment) {
		if(increment == 0)
			throw new IllegalArgumentException("increment cannot equal zero");
		if(start_inc > end_exc && increment > 0)
			throw new IllegalArgumentException("increment can't be positive for this range");
		if(start_inc < end_exc && increment < 0)
			throw new IllegalArgumentException("increment can't be negative for this range");
		
		
		int length = FastMath.abs(end_exc - start_inc);
		if(length == 0)
			throw new IllegalArgumentException("start_inc ("+start_inc+") cannot equal end_exc ("+end_exc+")");
		if(length%FastMath.abs(increment)!=0)
			throw new IllegalArgumentException("increment will not create evenly spaced elements");
		length /= FastMath.abs(increment);
		
		
		final int[] out = new int[length];
		if(increment < 0) {
			int j = 0;
			for(int i = start_inc; i > end_exc; i+=increment) out[j++] = i;
		} else {
			int j = 0;
			for(int i = start_inc; i < end_exc; i+=increment) out[j++] = i;
		}
		
		return out;
	}
	
	
	public static int argMax(final double[] v) {
		checkDims(v);
		
		double max = SIGNED_MIN;
		int max_idx = -1;
		
		for(int i = 0; i < v.length; i++) {
			double val = v[i];
			if(val > max) {
				max = val;
				max_idx = i;
			}
		}
		
		return max_idx;
	}
	
	public static int argMin(final double[] v) {
		checkDims(v);
		
		double min = MAX;
		int min_idx = -1;
		
		for(int i = 0; i < v.length; i++) {
			double val = v[i];
			if(val < min) {
				min = val;
				min_idx = i;
			}
		}
		
		return min_idx;
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
	
	public static double[] completeCases(final double[] d) {
		checkDims(d);
		final ArrayList<Double> out = new ArrayList<>();
		
		for(double dub: d)
			if(!Double.isNaN(dub))
				out.add(dub);
		
		final double[] copy = new double[out.size()];
		for(int i = 0; i < out.size(); i++)
			copy[i] = out.get(i);
		
		return copy;
	}
	
	
	/**
	 * Identifies whether a vector contains any missing values. 
	 * Depending on {@link GlobalState} parallelism settings, auto schedules parallel
	 * @param a
	 * @return true if vector contains any NaNs
	 */
	public static boolean containsNaN(final double[] a) {
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return containsNaNDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return containsNaNForceSerial(a);
	}
	
	
	/**
	 * Identifies whether a vector contains any missing values
	 * in a parallel fashion.
	 * @param a
	 * @return true if vector contains any NaNs
	 */
	public static boolean containsNaNDistributed(final double[] a) {
		return DistributedNaNCheck.containsNaN(a);
	}
	
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static boolean containsNaNForceSerial(final double[] a) {
		for(double b: a)
			if(Double.isNaN(b))
				return true;
		
		return false;
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
	
	/**
	 * Returns a shallow copy of the arg ArrayList. If the generic
	 * type is immutable (an instance of Number, String, etc) will
	 * act as a deep copy.
	 * @param a
	 * @return a shallow copy
	 */
	public static <T> ArrayList<T> copy(final ArrayList<T> a) {
		final ArrayList<T> copy = new ArrayList<T>(a.size());
		
		for(T i: a)
			copy.add(i);
		
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
	
	public static boolean equalsExactly(final int[] a, final int[] b) {
		checkDims(a, b);
		
		
		for(int i = 0; i < a.length; i++)
			if(a[i] != b[i])
				return false;
		return true;
	}
	
	/**
	 * Checks whether two vectors are exactly equal.
	 * Automatically assigns parallel or serial jobs depending 
	 * on settings in {@link GlobalState}
	 * @param a
	 * @param b
	 * @param eps
	 * @return whether the two vectors are exactly equal
	 */
	public static boolean equalsExactly(final double[] a, final double[] b) {
		checkDims(a, b);
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return equalsExactlyDistributed(a, b);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return equalsWithToleranceForceSerial(a, b, 0);
	}
	
	
	/**
	 * Checks whether two vectors are exactly equal 
	 * in a distributed fashion
	 * @param a
	 * @param b
	 * @param eps
	 * @return whether the two vectors are exactly equal
	 */
	public static boolean equalsExactlyDistributed(final double[] a, final double[] b) {
		return DistributedEqualityTest.equalsExactly(a, b);
	}
	
	
	/**
	 * Checks whether two vectors are equal within a 
	 * tolerance of {@link Precision#EPSILON}. Automatically assigns parallel
	 * or serial jobs depending on settings in {@link GlobalState}
	 * @param a
	 * @param b
	 * @param eps
	 * @return whether the two vectors are equal within a certain tolerance
	 */
	public static boolean equalsWithTolerance(final double[] a, final double[] b) {
		return equalsWithTolerance(a, b, Precision.EPSILON);
	}
	
	
	/**
	 * Checks whether two vectors are equal within a 
	 * specified tolerance. Automatically assigns parallel
	 * or serial jobs depending on settings in {@link GlobalState}
	 * @param a
	 * @param b
	 * @param eps
	 * @return whether the two vectors are equal within a certain tolerance
	 */
	public static boolean equalsWithTolerance(final double[] a, final double[] b, final double eps) {
		checkDims(a, b);
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return equalsWithToleranceDistributed(a, b, eps);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return equalsWithToleranceForceSerial(a, b, eps);
	}
	
	
	/**
	 * Checks whether two vectors are equal within a 
	 * specified tolerance in a distributed fashion
	 * @param a
	 * @param b
	 * @param eps
	 * @return whether the two vectors are equal within a certain tolerance
	 */
	public static boolean equalsWithToleranceDistributed(final double[] a, final double[] b, final double eps) {
		return DistributedEqualityTest.equalsWithTolerance(a, b, eps);
	}
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static boolean equalsWithToleranceForceSerial(final double[] a, final double[] b, final double eps) {
		for(int i = 0; i < a.length; i++)
			if( !Precision.equals(a[i], b[i], eps) )
				return false;
		
		return true;
	}
	
	/**
	 * Given a min value, <tt>min</tt>, any value in the input vector lower than the value
	 * will be truncated to another floor value, <tt>floor</tt>
	 * @param a
	 * @param min
	 * @param floor
	 * @return the truncated vector
	 */
	public static double[] floor(final double[] a, final double min, final double floor) {
		checkDims(a);
		
		final double[] b = new double[a.length];
		for(int i = 0; i < b.length; i++)
			b[i] = a[i] < min ? floor : a[i];
		
		return b;
	}
	
	/**
	 * Calculate the inner product between two vectors. If {@link GlobalState} allows
	 * for auto parallelism and the size of the vectors are greater than the max serial
	 * value alotted in GlobalState, will automatically schedule a parallel job.
	 * @param a
	 * @param b
	 * @return the inner product between a and b
	 */
	public static double innerProduct(final double[] a, final double[] b) {
		checkDims(a, b);
		if(ALLOW_PARALLELISM && a.length>MAX_SERIAL_VECTOR_LEN) {
			try {
				return innerProductDistributed(a, b);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return innerProductForceSerial(a, b);
	}
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static double innerProductForceSerial(final double[] a, final double[] b) {
		double sum = 0d;
		for(int i = 0; i < a.length; i++)
			sum += a[i] * b[i];
		
		return sum;
	}
	
	/**
	 * Calculate the inner product between a and b in a distributed fashion
	 * @param a
	 * @param b
	 * @return the inner product between a and b
	 */
	public static double innerProductDistributed(final double[] a, final double[] b) {
		return DistributedInnerProduct.innerProd(a, b);
	}
	
	public static double iqr(final double[] a) {
		checkDims(a);
		DescriptiveStatistics d = new DescriptiveStatistics(a);
		return d.getPercentile(75) - d.getPercentile(25);
	}
	
	public static boolean isOrthogonalTo(final double[] a, final double[] b) {
		// Will auto determine whether parallel is necessary or allowed...
		return Precision.equals(innerProduct(a, b), 0, Precision.EPSILON);
	}
	
	public static double kurtosis(final double[] a) {
		checkDims(a);
		DescriptiveStatistics d = new DescriptiveStatistics(a);
		return d.getKurtosis();
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
		return FastMath.pow(sum(pow(abs(a), p)), power);
	}
	
	public static double magnitude(final double[] a) {
		return l2Norm(a);
	}
	
	final public static double max(final double[] a) {
		double max = SIGNED_MIN;
		for(double d : a)
			if(d > max)
				max = d;
		return max;
	}
	
	final public static double mean(final double[] a) {
		return mean(a, sum(a));
	}
	
	final protected static double mean(final double[] a, final double sum) {
		return sum / a.length;
	}
	
	public static double median(final double[] a) {
		checkDims(a);
		if(a.length == 1)
			return a[0];
		
		// Get copy, sort it
		final double[] copy = copy(a);
		Arrays.sort(copy);
		
		int mid = copy.length/2;
		if(copy.length%2 != 0) // if not even in length
			return copy[mid];
		
		return (copy[mid-1]+copy[mid])/2d;
	}
	
	final public static double min(final double[] a) {
		double min = MAX;
		for(double d : a)
			if(d < min)
				min = d;
		return min;
	}
	
	
	/**
	 * Multiply each respective element from two vectors. Yields a vector of equal length.
	 * Auto selects parallelism or serialism depending on parallel settings in {@link GlobalState}
	 * @param a
	 * @param b
	 * @return the product of two vectors
	 */
	public static double[] multiply(final double[] a, final double[] b) {
		checkDims(a, b);
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return multiplyDistributed(a, b);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return multiplyForceSerial(a, b);
	}
	
	
	/**
	 * Multiply each respective element from two vectors in a parallel fashion. 
	 * Yields a vector of equal length.
	 * @param a
	 * @param b
	 * @return the product of two vectors
	 */
	public static double[] multiplyDistributed(final double[] a, final double[] b) {
		return DistributedMultiply.operate(a, b);
	}
	
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] multiplyForceSerial(final double[] a, final double[] b) {
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] * b[i];
		
		return ab;
	}
	
	
	/**
	 * Count the nans in a vector. Auto selects parallelism or serialism 
	 * depending on parallel settings in {@link GlobalState}
	 * @param a
	 * @return the number of nans in the vector
	 */
	public static int nanCount(final double[] a) {
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return nanCountDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return nanCountForceSerial(a);
	}
	
	/**
	 * Count the nans in a vector in a parallel fashion
	 * @param a
	 * @return the number of nans in the vector
	 */
	public static int nanCountDistributed(final double[] a) {
		return DistributedNaNCount.nanCount(a);
	}
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static int nanCountForceSerial(final double[] a) {
		int ct = 0;
		for(double d: a)
			if(Double.isNaN(d))
				ct++;
		
		return ct;
	}
	
	public static double nanMax(final double[] a) {
		checkDims(a);
		
		double max = GlobalState.Mathematics.SIGNED_MIN;
		for(double d: a) {
			if(Double.isNaN(d))
				continue;
			if(d > max)
				max = d;
		}
		
		return max == GlobalState.Mathematics.SIGNED_MIN ? Double.NaN : max;
	}
	
	public static double nanMin(final double[] a) {
		checkDims(a);
		
		double min = GlobalState.Mathematics.MAX;
		for(double d: a) {
			if(Double.isNaN(d))
				continue;
			if(d < min)
				min = d;
		}
		
		return min == GlobalState.Mathematics.MAX ? Double.NaN : min;
	}
	
	public static double nanMean(final double[] a) {
		double sum = 0;
		int count = 0;
		for(double d: a) {
			if(!Double.isNaN(d)) {
				count++;
				sum += d;
			}
		}
		
		return count == 0 ? Double.NaN : sum / (double)count;
	}
	
	public static double nanMedian(final double[] a) {
		return median(completeCases(a));
	}
	
	final public static double nanStdDev(final double[] a) {
		return nanStdDev(a, DEF_SUBTRACT_ONE_VAR);
	}
	
	public final static double nanStdDev(final double[] a, final double mean) {
		return nanStdDev(a, mean, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double nanStdDev(final double[] a, final boolean n_minus_one) {
		return nanStdDev(a, nanMean(a), n_minus_one);
	}
	
	final protected static double nanStdDev(final double[] a, final double mean, final boolean n_minus_one) {
		return FastMath.sqrt(nanVar(a, mean, n_minus_one));
	}
	
	public static double nanSum(final double[] a) {
		double sum = 0;
		for(double d: a)
			if(!Double.isNaN(d))
				sum += d;
		
		return sum;
	}
	
	final public static double nanVar(final double[] a) {
		return nanVar(a, DEF_SUBTRACT_ONE_VAR);
	}
	
	final protected static double nanVar(final double[] a, final double mean) {
		return nanVar(a, mean, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double nanVar(final double[] a, final boolean n_minus_one) {
		return nanVar(a, nanMean(a), n_minus_one);
	}
	
	final protected static double nanVar(final double[] a, final double mean, final boolean n_minus_one) {
		if(Double.isNaN(mean)) // Here we already know the whole thing is NaN
			return mean;
		
		boolean seenNonNan = false;
		double sum = 0;
		for(double x : a) {
			if(Double.isNaN(x))
				continue;
			
			seenNonNan = true;
			double res = x - mean; // Want to avoid math.pow...
			sum += res * res;
		}
		
		return !seenNonNan ? Double.NaN : sum / (a.length - (n_minus_one ? 1 : 0));
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
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return prodDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return prodForceSerial(a);
	}
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static double prodForceSerial(final double[] a) {
		double prod = 1;
		for(double d: a)
			prod *= d;
		return prod;
	}
	
	public static double prodDistributed(final double[] a) {
		return DistributedProduct.prod(a);
	}
	
	public static double[] randomGaussian(final int n) {
		return randomGaussian(n, new Random());
	}
	
	
	public static double[] randomGaussian(final int n, final Random seed) {
		return randomGaussian(n,seed,1);
	}
	
	public static double[] randomGaussian(final int n, final double scalar) {
		return randomGaussian(n, new Random(), scalar);
	}
	
	
	public static double[] randomGaussian(final int n, final Random seed, final double scalar) {
		if(n < 1)
			throw new IllegalArgumentException("illegal dimensions");
		
		final double[] out = new double[n];
		for(int i = 0; i < n; i++)
			out[i] = seed.nextGaussian()*scalar;
		
		return out;
	}
	
	public static double[] randomGaussianNoiseVector(final int n) {
		return randomGaussian(n, new Random());
	}
	
	public static double[] randomGaussianNoiseVector(final int n, final Random seed) {
		return randomGaussian(n,seed,GlobalState.Mathematics.EPS);
	}
	
	/**
	 * Create a vector of a repeated value
	 * @param val
	 * @param n
	 * @return a vector of a repeated value
	 */
	public static double[] rep(final double val, final int n) {
		if(n < 0)
			throw new IllegalArgumentException(n+" must not be negative");
		final double[] d = new double[n];
		for(int i = 0; i < n; i++)
			d[i] = val;
		return d;
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
	
	public final static double stdDev(final double[] a, final double mean) {
		return stdDev(a, mean, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double stdDev(final double[] a, final boolean n_minus_one) {
		return stdDev(a, mean(a), n_minus_one);
	}
	
	final protected static double stdDev(final double[] a, final double mean, final boolean n_minus_one) {
		return FastMath.sqrt(var(a, mean, n_minus_one));
	}
	
	public static double[] subtract(final double[] from, final double[] subtractor) {
		checkDims(from, subtractor);
		
		final double[] ab = new double[from.length];
		for(int i = 0; i < from.length; i++)
			ab[i] = from[i] - subtractor[i];
		
		return ab;
	}
	
	public static double sum(final double[] a) {
		if(ALLOW_PARALLELISM && null!=a && a.length > MAX_SERIAL_VECTOR_LEN) {
			try {
				return sumDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
			
		return sumForceSerial(a);
	}
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @param b
	 * @return
	 */
	public static double sumForceSerial(final double[] a) {
		double sum = 0d;
		for(double d : a)
			sum += d;
		return sum;
	}
	
	public static double sumDistributed(final double[] a) {
		return DistributedSum.sum(a);
	}
	
	
	
	/**
	 * Returns the count of <tt>true</tt> in a boolean vector
	 * @param a
	 * @return
	 */
	public static int sum(final boolean[] a) {
		int sum = 0;
		for(boolean b: a)
			if(b) sum++;
		return sum;
	}
	
	public static <T> LinkedHashSet<T> unique(final T[] arr) {
		final LinkedHashSet<T> out = new LinkedHashSet<>();
		for(T t: arr)
			out.add(t);
		return out;
	}
	
	final public static double var(final double[] a) {
		return var(a, DEF_SUBTRACT_ONE_VAR);
	}
	
	final protected static double var(final double[] a, final double mean) {
		return var(a, mean, DEF_SUBTRACT_ONE_VAR);
	}
	
	final public static double var(final double[] a, final boolean n_minus_one) {
		return var(a, mean(a), n_minus_one);
	}
	
	final protected static double var(final double[] a, final double mean, final boolean n_minus_one) {
		double sum = 0;
		for(double x : a) {
			double res = x - mean; // Want to avoid math.pow...
			sum += res * res;
		}
		return sum / (a.length - (n_minus_one ? 1 : 0));
	}
}
