package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

import com.clust4j.GlobalState;
import com.clust4j.utils.parallel.map.DistributedAbs;
import com.clust4j.utils.parallel.map.DistributedAdd;
import com.clust4j.utils.parallel.map.DistributedLog;
import com.clust4j.utils.parallel.map.DistributedMultiply;
import com.clust4j.utils.parallel.map.DistributedSubtract;
import com.clust4j.utils.parallel.reduce.DistributedEqualityTest;
import com.clust4j.utils.parallel.reduce.DistributedInnerProduct;
import com.clust4j.utils.parallel.reduce.DistributedNaNCheck;
import com.clust4j.utils.parallel.reduce.DistributedNaNCount;
import com.clust4j.utils.parallel.reduce.DistributedProduct;
import com.clust4j.utils.parallel.reduce.DistributedSum;

import static com.clust4j.GlobalState.ParallelismConf.MAX_SERIAL_VECTOR_LEN;
import static com.clust4j.GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM;
import static com.clust4j.GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE;
import static com.clust4j.GlobalState.Mathematics.MAX;
import static com.clust4j.GlobalState.Mathematics.SIGNED_MIN;

public class VecUtils {
	final static String VEC_LEN_ERR = "illegal vector length: ";
	public final static int MIN_ACCEPTABLE_VEC_LEN = 1;
	public final static boolean DEF_SUBTRACT_ONE_VAR = true;
	
	
	
	/**
	 * Create a boolean vector
	 * @author Taylor G Smith
	 */
	public static class VecSeries extends Series<boolean[]> {
		final boolean[] vec;
		final int n;
		
		
		private VecSeries(double[] v) {
			checkDims(v);
			this.n = v.length;
			this.vec = new boolean[n];
		}
		
		public VecSeries(double[] v, Inequality in, double val) {
			this(v);
			for(int j = 0; j < n; j++)
				vec[j] = eval(v[j], in, val);
		}
		
		public VecSeries(double[] a, Inequality in, double[] b) {
			this(a);
			if(n != b.length)
				throw new DimensionMismatchException(n, b.length);
			
			for(int i = 0; i < n; i++)
				vec[i] = eval(a[i], in, b[i]);
		}
		
		@Override
		public boolean[] getRef() {
			return vec;
		}
		
		@Override
		public boolean[] get() {
			return copy(vec);
		}
	}
	
	
	
	
	
	
	
	// =============== DIM CHECKS ==============
	final private static void dimAssess(final int a) { if(a < MIN_ACCEPTABLE_VEC_LEN) throw new IllegalArgumentException(VEC_LEN_ERR + a); }
	final static public void checkDims(final byte[] a) 		{ dimAssess(a.length); }
	final static public void checkDims(final short[] a) 	{ dimAssess(a.length); }
	final static public void checkDims(final boolean[] a) 	{ dimAssess(a.length); }
	final static public void checkDims(final int[] a) 		{ dimAssess(a.length); }
	final static public void checkDims(final float[] a) 	{ dimAssess(a.length); }
	final static public void checkDims(final double[] a) 	{ dimAssess(a.length); }
	final static public void checkDims(final long[] a) 		{ dimAssess(a.length); }
	
	final private static void dimAssessPermitEmpty(final int a) 	{ if(a < 0) throw new IllegalArgumentException(VEC_LEN_ERR + a); }
	final static public void checkDimsPermitEmpty(final byte[] a) 	{ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final short[] a) 	{ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final boolean[] a){ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final int[] a) 	{ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final float[] a) 	{ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final double[] a) { dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final long[] a) 	{ dimAssessPermitEmpty(a.length); }
	
	final private static void dimAssess(final int a, final int b) { if(a != b) throw new DimensionMismatchException(a, b); dimAssess(a); }
	final static public void checkDims(final byte[] a, final byte[] b) 		{ dimAssess(a.length, b.length); }
	final static public void checkDims(final short[] a, final short[] b) 	{ dimAssess(a.length, b.length); }
	final static public void checkDims(final boolean[] a, final boolean[] b){ dimAssess(a.length, b.length); }
	final static public void checkDims(final int[] a, final int[] b) 		{ dimAssess(a.length, b.length); }
	final static public void checkDims(final float[] a, final float[] b) 	{ dimAssess(a.length, b.length); }
	final static public void checkDims(final double[] a, final double[] b) 	{ dimAssess(a.length, b.length); }
	final static public void checkDims(final long[] a, final long[] b) 		{ dimAssess(a.length, b.length); }
	
	
	
	
	
	
	
	
	
	// ====================== MATH FUNCTIONS =======================
	/**
	 * Calculate the absolute value of the values in the vector and return a copy.
	 * Depending on {@link GlobalState} parallelism settings, auto schedules parallel
	 * or serial job.
	 * @param a
	 * @return absolute value of the vector
	 */
	public static double[] abs(final double[] a) {
		checkDims(a);
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
	public static double[] absForceSerial(final double[] a) {
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
	public static double[] addForceSerial(final double[] a, final double[] b) {
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
	
	public static int[] argSort(final double[] a) {
		checkDims(a);
		
		final int n = a.length;
		final TreeMap<Double, Integer> map = new TreeMap<>();
		for(int i = 0; i < n; i++)
			map.put(a[i], i);
		
		int idx = 0;
		final int[] res = new int[n];
		for(Map.Entry<Double,Integer> entry: map.entrySet())
			res[idx++] = entry.getValue();
		
		return res;
	}
	
	public static int[] argSort(final int[] a) {
		checkDims(a);
		
		final int n = a.length;
		final TreeMap<Integer, Integer> map = new TreeMap<>();
		for(int i = 0; i < n; i++)
			map.put(a[i], i);
		
		int idx = 0;
		final int[] res = new int[n];
		for(Map.Entry<Integer,Integer> entry: map.entrySet())
			res[idx++] = entry.getValue();
		
		return res;
	}
	
	public static double[] asDouble(final int[] a) {
		checkDims(a);
		final int n = a.length;
		final double[] d = new double[n];
		
		for(int i = 0; i < n; i++)
			d[i] = (double)a[i];
		
		return d;
	}
	
	final public static int[] cat(final int[] a, final int[] b) {
		checkDims(a);
		checkDims(b);
		
		final int na = a.length, nb = b.length, n = na+nb;
		if(na == 0) return copy(b);
		if(nb == 0) return copy(a);
		
		final int[] res = new int[n];
		for(int i = 0; i < na; i++)
			res[i] = a[i];
		for(int i = 0; i < nb; i++)
			res[i+na] = b[i];
		
		return res;
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
		return DistributedNaNCheck.operate(a);
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
	
	public static boolean[] copy(final boolean[] b) {
		final boolean[] copy = new boolean[b.length];
		System.arraycopy(b, 0, copy, 0, b.length);
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
	
	public static String[] copy(final String[] s) {
		final String[] copy = new String[s.length];
		System.arraycopy(s, 0, copy, 0, s.length);
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
		return DistributedEqualityTest.operate(a, b, 0);
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
		return DistributedEqualityTest.operate(a, b, eps);
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
	
	public static double[] exp(final double[] a) {
		checkDims(a);
		
		final int n = a.length;
		final double[] out = new double[n];
		for(int i = 0; i < n; i++)
			out[i] = FastMath.exp(a[i]);
		
		return out;
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && 
				 a.length>MAX_SERIAL_VECTOR_LEN)) {
			try {
				return innerProductDistributed(a, b);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return innerProductForceSerial(a, b);
	}
	
	/**
	 * Calculate the inner product between a and b in a distributed fashion
	 * @param a
	 * @param b
	 * @return the inner product between a and b
	 */
	public static double innerProductDistributed(final double[] a, final double[] b) {
		return DistributedInnerProduct.operate(a, b);
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
	
	/**
	 * Calculate the log of the vector. If {@link GlobalState} allows
	 * for auto parallelism and the size of the vectors are greater than the max serial
	 * value alotted in GlobalState, will automatically schedule a parallel job.
	 * @param a
	 * @return the log of the vector
	 */
	public static double[] log(final double[] a) {
		checkDims(a);
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && 
				 a.length>MAX_SERIAL_VECTOR_LEN)) {
			try {
				return logDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
		
		return logForceSerial(a);
	}
	
	
	/**
	 * Calculate the log of the vector in a parallel fashion
	 * @param a
	 * @return the log of the vector
	 */
	public static double[] logDistributed(final double[] a) {
		return DistributedLog.operate(a);
	}
	
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param a
	 * @return
	 */
	public static double[] logForceSerial(final double[] a) {
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
	
	/**
	 * Calculates the l2 norm of the vector
	 * @param a
	 * @return the vector magnitude
	 */
	public static double magnitude(final double[] a) {
		return l2Norm(a);
	}
	
	/**
	 * Identify the max value in the vector
	 * @param a
	 * @return the max in the vector
	 */
	final public static double max(final double[] a) {
		double max = SIGNED_MIN;
		for(double d : a)
			if(d > max)
				max = d;
		return max;
	}
	
	/**
	 * Calculate the mean of the vector
	 * @param a
	 * @return the mean of the vector
	 */
	final public static double mean(final double[] a) {
		return mean(a, sum(a));
	}
	
	/**
	 * Calculate the mean of the vector, given its sum
	 * @param a
	 * @param sum
	 * @return the mean of the vector
	 */
	final protected static double mean(final double[] a, final double sum) {
		return sum / a.length;
	}
	
	/**
	 * Calculate the median of the vector
	 * @param a
	 * @return the vector median
	 */
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
	
	/**
	 * Identify the min value in the vector
	 * @param a
	 * @return the min in the vector
	 */
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
		return DistributedNaNCount.operate(a);
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
	
	/**
	 * Identify the max value in the vector, excluding NaNs
	 * @param a
	 * @return the max in the vector
	 */
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
	
	/**
	 * Identify the min value in the vector, excluding NaNs
	 * @param a
	 * @return the min in the vector
	 */
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
	
	/**
	 * Calculate the mean of the vector, excluding NaNs
	 * @param a
	 * @return the mean of the vector
	 */
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
	
	/**
	 * Calculate the median of the vector, excluding NaNs
	 * @param a
	 * @return the median of the vector
	 */
	public static double nanMedian(final double[] a) {
		return median(completeCases(a));
	}
	
	/**
	 * Calculate the standard deviation of the vector, excluding NaNs
	 * @param a
	 * @return the std dev of the vector
	 */
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
	 * Adapted from Numpy's partition method.
	 * Creates a copy of the array with its elements 
	 * rearranged in such a way that the value of the 
	 * element in kth position is in the position it would 
	 * be in a sorted array. All elements smaller than the 
	 * kth element are moved before this element and all equal 
	 * or greater are moved behind it. The ordering of the elements 
	 * in the two partitions is undefined.
	 * @param a
	 * @param kth
	 * @return
	 */
	public static double[] partition(final double[] a, final int kth) {
		checkDims(a);
		final int n = a.length;
		if(kth >= n || kth < 0)
			throw new IllegalArgumentException(kth+" is out of bounds");
		
		final double val = a[kth];
		double[] b = VecUtils.copy(a);
		double[] c = new double[n];
		
		int idx = -1;
		Arrays.sort(b);
		for(int i = 0; i < n; i++) {
			if(b[i] == val) {
				idx = i;
				break;
			}	
		}
		
		c[idx] = val;
		for(int i = 0, nextLow = 0, nextHigh = idx+1; i < n; i++) {
			if(i == kth) // This is the pivot point
				continue;
			if(a[i] < val)
				c[nextLow++] = a[i];
			else c[nextHigh++] = a[i];
		}
		
		return c;
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
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
		return DistributedProduct.operate(a);
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
	
	public static double[] reorder(final double[] data, final int[] order) {
		VecUtils.checkDims(order);
		VecUtils.checkDims(data);
		
		final int n = order.length;
		final double[] out = new double[n];
		
		int idx = 0;
		for(int i: order)
			out[idx++] = data[i];
		
		return out;
	}
	
	public static int[] reorder(final int[] data, final int[] order) {
		VecUtils.checkDims(order);
		VecUtils.checkDims(data);
		
		final int n = order.length;
		final int[] out = new int[n];
		
		int idx = 0;
		for(int i: order)
			out[idx++] = data[i];
		
		return out;
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
	
	/**
	 * Create a vector of a repeated value
	 * @param val
	 * @param n
	 * @return a vector of a repeated value
	 */
	public static int[] repInt(final int val, final int n) {
		if(n < 0)
			throw new IllegalArgumentException(n+" must not be negative");
		final int[] d = new int[n];
		for(int i = 0; i < n; i++)
			d[i] = val;
		return d;
	}
	
	/**
	 * Create a vector of a repeated value
	 * @param val
	 * @param n
	 * @return a vector of a repeated value
	 */
	public static boolean[] repBool(final boolean val, final int n) {
		if(n < 0)
			throw new IllegalArgumentException(n+" must not be negative");
		final boolean[] d = new boolean[n];
		for(int i = 0; i < n; i++)
			d[i] = val;
		return d;
	}
	
	public static double[] reverseSeries(final double[] a) {
		checkDims(a);
		
		final int n = a.length;
		final double[] out = new double[n];
		for(int i = n - 1, j = 0; i >= 0; i--, j++)
			out[j] = a[i];
		
		return out;
	}
	
	public static int[] reverseSeries(final int[] a) {
		checkDims(a);
		
		final int n = a.length;
		final int[] out = new int[n];
		for(int i = n - 1, j = 0; i >= 0; i--, j++)
			out[j] = a[i];
		
		return out;
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
	
	public static double[] slice(final double[] a, final int startInc, final int endExc) {
		checkDims(a);
		
		if(endExc > a.length)
			throw new ArrayIndexOutOfBoundsException(endExc);
		if(startInc < 0 || startInc > a.length || startInc >= endExc)
			throw new ArrayIndexOutOfBoundsException(startInc);
		
		final double[] out = new double[endExc - startInc];
		for(int i = startInc, j = 0; i < endExc; i++, j++)
			out[j] = a[i];
		
		return out;
	}
	
	public static int[] slice(final int[] a, final int startInc, final int endExc) {
		checkDims(a);
		
		if(endExc > a.length)
			throw new ArrayIndexOutOfBoundsException(endExc);
		if(startInc < 0 || startInc > a.length || startInc >= endExc)
			throw new ArrayIndexOutOfBoundsException(startInc);
		
		final int[] out = new int[endExc - startInc];
		for(int i = startInc, j = 0; i < endExc; i++, j++)
			out[j] = a[i];
		
		return out;
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
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=from && 
				 from.length > MAX_SERIAL_VECTOR_LEN)) {
			try {
				return subtractDistributed(from, subtractor);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
			
		return subtractForceSerial(from, subtractor);
	}
	
	
	public static double[] subtractDistributed(final double[] from, final double[] subtractor) {
		return DistributedSubtract.operate(from, subtractor);
	}
	
	/**
	 * If another parallelized operation is calling this one, we should force this
	 * one to be run serially so as not to inundate the cores with multiple recursive tasks.
	 * @param from
	 * @param subtractor
	 * @return
	 */
	public static double[] subtractForceSerial(final double[] from, final double[] subtractor) {
		checkDims(from, subtractor);
		
		final double[] ab = new double[from.length];
		for(int i = 0; i < from.length; i++)
			ab[i] = from[i] - subtractor[i];
		
		return ab;
	}
	
	public static double sum(final double[] a) {
		if(FORCE_PARALLELISM_WHERE_POSSIBLE || 
				(ALLOW_AUTO_PARALLELISM && null!=a && 
				 a.length > MAX_SERIAL_VECTOR_LEN)) {
			try {
				return sumDistributed(a);
			} catch(RejectedExecutionException e) { /*Perform normal execution*/ }
		}
			
		return sumForceSerial(a);
	}
	
	public static double sumDistributed(final double[] a) {
		return DistributedSum.operate(a);
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
	
	public static double[][] vstack(final double[] a, final double[] b) {
		checkDims(a,b);
		
		final int n = a.length;
		final double[][] out = new double[2][n];
		
		out[0] = copy(a);
		out[1] = copy(b);
		
		return out;
	}
	
	public static int[][] vstack(final int[] a, final int[] b) {
		checkDims(a,b);
		
		final int n = a.length;
		final int[][] out = new int[2][n];
		
		out[0] = copy(a);
		out[1] = copy(b);
		
		return out;
	}
	
	public static double[] where(final VecSeries series, final double[] x, final double[] y) {
		checkDims(x,y);
		
		final int n = x.length;
		final boolean[] ser = series.get();
		if(ser.length != n)
			throw new DimensionMismatchException(ser.length, n);
			
		final double[] result = new double[n];
		for(int i = 0; i < n; i++)
			result[i] = ser[i] ? x[i] : y[i];
			
		return result;
	}
	
	public static double[] where(final VecSeries series, final double x, final double[] y) {
		return where(series, rep(x, series.get().length), y);
	}
	
	public static double[] where(final VecSeries series, final double[] x, final double y) {
		return where(series, x, rep(y, series.get().length));
	}
	
	public static double[] where(final VecSeries series, final double x, final double y) {
		return where(series, rep(x,series.get().length), rep(y,series.get().length));
	}
}
