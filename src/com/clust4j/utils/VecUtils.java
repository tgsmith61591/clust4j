package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.concurrent.RecursiveTask;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

public class VecUtils {
	/** Double.MIN_VALUE is not negative; this is */
	public final static double SAFE_MIN = Double.NEGATIVE_INFINITY;
	public final static double SAFE_MAX = Double.POSITIVE_INFINITY;
	private final static int MAX_DIST_LEN = 10_000_000;
	public final static int MIN_ACCEPTABLE_VEC_LEN = 1;
	public final static boolean DEF_SUBTRACT_ONE_VAR = true;
	
	/** If true and the size of the vector exceeds {@value #MAX_DIST_LEN}, 
	 *  auto schedules parallel job for applicable operations */
	public static boolean ALLOW_AUTO_PARALLELISM = true;
	
	
	
	
	
	
	// ================================== DISTRIBUTED CLASSES ================================
	static abstract private class DistributedVectorTask<T> extends RecursiveTask<T> {
		private static final long serialVersionUID = -7986981765361158408L;
		static final int SEQUENTIAL_THRESHOLD = 2_500_000;

	    final double[] array;
		final int low;
	    final int high;
		
		DistributedVectorTask(double[] arr, int lo, int hi) {
			array = arr;
			low = lo;
			high = hi;
		}
	}
	
	/**
	 * A class for distributed NaN checks
	 * @author Taylor G Smith
	 */
	static private class NaNCheckDistributor extends DistributedVectorTask<Boolean> {
		private static final long serialVersionUID = -4107497709587691394L;

		NaNCheckDistributor(final double[] arr, int lo, int hi) {
			super(arr, lo, hi);
		}

		@Override
		protected Boolean compute() {
			if(high - low <= SEQUENTIAL_THRESHOLD) {
	            for(int i=low; i < high; ++i)
	                if(Double.isNaN(array[i]))
	            		return true;
	            
	            return false;
	         } else {
	            int mid = low + (high - low) / 2;
	            NaNCheckDistributor left  = new NaNCheckDistributor(array, low, mid);
	            NaNCheckDistributor right = new NaNCheckDistributor(array, mid, high);
	            left.fork();
	            boolean rightAns = right.compute();
	            boolean leftAns  = left.join();
	            return leftAns || rightAns;
	         }
		}
		
		public static boolean containsNaN(final double[] array) {
			return ConcurrencyUtils.fjPool.invoke(new NaNCheckDistributor(array,0,array.length));
		}
	}
	
	/**
	 * A base class for distributed vector operations
	 * @author Taylor G Smith
	 */
	static abstract private class DistributedOperator extends DistributedVectorTask<Double> {
		private static final long serialVersionUID = 704439933447978232L;
		
		DistributedOperator(final double[] arr, int lo, int hi) {
	        super(arr, lo, hi);
	    }
	}
	
	final private static class NaNCountDistributor extends DistributedVectorTask<Integer> {
		private static final long serialVersionUID = 5031788548523204436L;

		private NaNCountDistributor(final double[] arr, int lo, int hi) {
			super(arr, lo, hi);
		}

		@Override
		protected Integer compute() {
			if(high - low <= SEQUENTIAL_THRESHOLD) {
	            int sum = 0;
	            for(int i=low; i < high; ++i) 
	                if(Double.isNaN(array[i]))
	                	sum++;
	            return sum;
	         } else {
	            int mid = low + (high - low) / 2;
	            NaNCountDistributor left  = new NaNCountDistributor(array, low, mid);
	            NaNCountDistributor right = new NaNCountDistributor(array, mid, high);
	            left.fork();
	            int rightAns = right.compute();
	            int leftAns  = left.join();
	            return leftAns + rightAns;
	         }
		}
		
		public static int nanCount(double[] array) {
			if(array.length == 0)
				return 0;
			return ConcurrencyUtils.fjPool.invoke(new NaNCountDistributor(array,0,array.length));
		}
	}
	
	/**
	 * A class for distributed products of vectors
	 * @author Taylor G Smith
	 */
	final private static class ProdDistributor extends DistributedOperator {
		private static final long serialVersionUID = -1038455192192012983L;

		private ProdDistributor(final double[] arr, int lo, int hi) {
			super(arr, lo, hi);
		}
		
		@Override
		protected Double compute() {
			if(high - low <= SEQUENTIAL_THRESHOLD) {
	            double prod = 1;
	            for(int i=low; i < high; ++i) 
	                prod *= array[i];
	            return prod;
	         } else {
	            int mid = low + (high - low) / 2;
	            ProdDistributor left  = new ProdDistributor(array, low, mid);
	            ProdDistributor right = new ProdDistributor(array, mid, high);
	            left.fork();
	            double rightAns = right.compute();
	            double leftAns  = left.join();
	            return leftAns * rightAns;
	         }
		}
		
		public static double prod(final double[] array) {
			checkDims(array);
			return ConcurrencyUtils.fjPool.invoke(new ProdDistributor(array,0,array.length));
		}
	}
	
	/**
	 * A class for distributed inner products of vectors
	 * @author Taylor G Smith
	 */
	final private static class InnerProdDistributor extends DistributedOperator {
		private static final long serialVersionUID = 9189105909360824409L;
		final double[] array_b;

	    private InnerProdDistributor(final double[] a, final double[] b, int lo, int hi) {
	        super(a, lo, hi);
	        array_b = b;
	    }

	    protected Double compute() {
	        if(high - low <= SEQUENTIAL_THRESHOLD) {
	            double sum = 0;
	            for(int i=low; i < high; ++i) 
	                sum += array[i] * array_b[i];
	            return sum;
	         } else {
	            int mid = low + (high - low) / 2;
	            InnerProdDistributor left  = new InnerProdDistributor(array, array_b, low, mid);
	            InnerProdDistributor right = new InnerProdDistributor(array, array_b, mid, high);
	            left.fork();
	            double rightAns = right.compute();
	            double leftAns  = left.join();
	            return leftAns + rightAns;
	         }
	     }

	     public static double innerProd(final double[] array, final double[] array_b) {
	    	 checkDims(array, array_b);
	         return ConcurrencyUtils.fjPool.invoke(new InnerProdDistributor(array,array_b,0,array.length));
	     }
	}
	
	/**
	 * A class for distributed summing of vectors
	 * @author Taylor G Smith
	 */
	final private static class SumDistributor extends DistributedOperator {
		private static final long serialVersionUID = -6086182277529660733L;

	    private SumDistributor(final double[] arr, int lo, int hi) {
	        super(arr, lo, hi);
	    }

	    protected Double compute() {
	        if(high - low <= SEQUENTIAL_THRESHOLD) {
	            double sum = 0;
	            for(int i=low; i < high; ++i) 
	                sum += array[i];
	            return sum;
	         } else {
	            int mid = low + (high - low) / 2;
	            SumDistributor left  = new SumDistributor(array, low, mid);
	            SumDistributor right = new SumDistributor(array, mid, high);
	            left.fork();
	            double rightAns = right.compute();
	            double leftAns  = left.join();
	            return leftAns + rightAns;
	         }
	     }

	     public static double sum(final double[] array) {
	    	 if(array.length == 0)
	    		 return 0;
	         return ConcurrencyUtils.fjPool.invoke(new SumDistributor(array,0,array.length));
	     }
	}
	// ===========================================================================================
	
	
	
	
	
	
	
	
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
		
		double max = SAFE_MIN;
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
		
		double min = SAFE_MAX;
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
	
	public static boolean containsNaN(final double[] a) {
		if(ALLOW_AUTO_PARALLELISM && null!=a && a.length > MAX_DIST_LEN)
			return containsNaNDistributed(a);
		
		for(double b: a)
			if(Double.isNaN(b))
				return true;
		
		return false;
	}
	
	public static boolean containsNaNDistributed(final double[] a) {
		return NaNCheckDistributor.containsNaN(a);
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
	
	public static ArrayList<Integer> copy(final ArrayList<Integer> a) {
		final ArrayList<Integer> copy = new ArrayList<Integer>(a.size());
		
		for(Integer i: a)
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
	
	public static double[] floor(final double[] a, final double min, final double floor) {
		checkDims(a);
		
		final double[] b = new double[a.length];
		for(int i = 0; i < b.length; i++)
			b[i] = a[i] < min ? floor : a[i];
		
		return b;
	}
	
	public static double innerProduct(final double[] a, final double[] b) {
		checkDims(a, b);
		if(ALLOW_AUTO_PARALLELISM && a.length>MAX_DIST_LEN)
			return innerProductDistributed(a, b);
		
		double sum = 0d;
		for(int i = 0; i < a.length; i++)
			sum += a[i] * b[i];
		
		return sum;
	}
	
	public static double innerProductDistributed(final double[] a, final double[] b) {
		return InnerProdDistributor.innerProd(a, b);
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
		return FastMath.pow(sum(pow(abs(a), p)), power);
	}
	
	public static double magnitude(final double[] a) {
		return l2Norm(a);
	}
	
	final public static double max(final double[] a) {
		double max = SAFE_MIN;
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
		double min = SAFE_MAX;
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
	
	public static int nanCount(final double[] a) {
		if(ALLOW_AUTO_PARALLELISM && null!=a && a.length > MAX_DIST_LEN)
			return nanCountDistributed(a);
		
		int ct = 0;
		for(double d: a)
			if(Double.isNaN(d))
				ct++;
		
		return ct;
	}
	
	public static int nanCountDistributed(final double[] a) {
		return NaNCountDistributor.nanCount(a);
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
		
		if(count == a.length)
			throw new NaNException("completely NaN vector");
		
		return sum / (double)count;
	}
	
	public static double nanMedian(final double[] a) {
		return median(completeCases(a));
	}
	
	public static double nanSum(final double[] a) {
		double sum = 0;
		for(double d: a)
			if(!Double.isNaN(d))
				sum += d;
		
		return sum;
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
		if(ALLOW_AUTO_PARALLELISM && null!=a && a.length > MAX_DIST_LEN)
			return prodDistributed(a);
		
		double prod = 1;
		for(double d: a)
			prod *= d;
		return prod;
	}
	
	public static double prodDistributed(final double[] a) {
		return ProdDistributor.prod(a);
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
		return randomGaussian(n,seed,MatUtils.EPS);
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
		if(ALLOW_AUTO_PARALLELISM && null!=a && a.length > MAX_DIST_LEN)
			return sumDistributed(a);
		
		double sum = 0d;
		for(double d : a)
			sum += d;
		return sum;
	}
	
	public static double sumDistributed(final double[] a) {
		return SumDistributor.sum(a);
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
