/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;

import com.clust4j.GlobalState;

import static com.clust4j.GlobalState.Mathematics.MAX;
import static com.clust4j.GlobalState.Mathematics.SIGNED_MIN;

public abstract class VecUtils {
	final static String VEC_LEN_ERR = "illegal vector length: ";
	public final static int MIN_ACCEPTABLE_VEC_LEN = 1;
	public final static boolean DEF_SUBTRACT_ONE_VAR = true;
	
	
	/**
	 * Abstract implementation of a vector series
	 * @author Taylor G Smith
	 */
	abstract static class VecSeries extends Series<boolean[]> {
		final boolean[] vec;
		final int n;
		
		/**
		 * Private constructor
		 * @throws IllegalArgumentException if the vector is empty
		 * @param v
		 */
		VecSeries(int v, Inequality in) {
			super(in);
			dimAssess(v);
			this.n = v;
			this.vec = new boolean[n];
		}
		
		@Override
		public boolean[] getRef() {
			return vec;
		}
		
		@Override
		public boolean[] get() {
			return copy(vec);
		}

		@Override
		public boolean all() {
			for(boolean b: vec)
				if(!b)
					return false;
			return true;
		}

		@Override
		public boolean any() {
			for(boolean b: vec)
				if(b)
					return true;
			return false;
		}
	}
	
	
	/**
	 * Create a boolean masking vector to be used in the 
	 * {@link VecUtils#where(DoubleSeries, double, double)} family
	 * of methods.
	 * @throws IllegalArgumentException if the input vector is empty
	 * @throws DimensionMismatchException if the input vector dims do not match
	 * @author Taylor G Smith
	 */
	public static class DoubleSeries extends VecSeries {
		
		/**
		 * Private constructor
		 * @throws IllegalArgumentException if the vector is empty
		 * @param v
		 */
		private DoubleSeries(double[] v, Inequality in) {
			super(v.length, in);
		}
		
		/**
		 * One vector constructor. Elements in the vector to the provided val
		 * @param a
		 * @param in
		 * @param val
		 * @throws IllegalArgumentException if the vector is empty
		 */
		public DoubleSeries(double[] v, Inequality in, double val) {
			this(v, in);
			for(int j = 0; j < n; j++)
				vec[j] = eval(v[j], val);
		}
		
		/**
		 * Two vector constructor. Compares respective elements.
		 * @param a
		 * @param in
		 * @param b
		 * @throws DimensionMismatchException if the dims of A and B don't match
		 * @throws IllegalArgumentException if the vector is empty
		 */
		public DoubleSeries(double[] a, Inequality in, double[] b) {
			this(a, in);
			if(n != b.length)
				throw new DimensionMismatchException(n, b.length);
			
			for(int i = 0; i < n; i++)
				vec[i] = eval(a[i], b[i]);
		}
	}
	
	
	
	/**
	 * Create a boolean masking vector wrapper
	 * @throws IllegalArgumentException if the input vector is empty
	 * @throws DimensionMismatchException if the input vector dims do not match
	 * @author Taylor G Smith
	 */
	public static class IntSeries extends VecSeries {
		
		/**
		 * Private constructor
		 * @throws IllegalArgumentException if the vector is empty
		 * @param v
		 */
		private IntSeries(int[] v, Inequality in) {
			super(v.length, in);
		}
		
		/**
		 * One vector constructor. Elements in the vector to the provided val
		 * @param a
		 * @param in
		 * @param val
		 * @throws IllegalArgumentException if the vector is empty
		 */
		public IntSeries(int[] v, Inequality in, int val) {
			this(v, in);
			for(int j = 0; j < n; j++)
				vec[j] = eval(v[j], val);
		}
		
		/**
		 * Two vector constructor. Compares respective elements.
		 * @param a
		 * @param in
		 * @param b
		 * @throws DimensionMismatchException if the dims of A and B don't match
		 * @throws IllegalArgumentException if the vector is empty
		 */
		public IntSeries(int[] a, Inequality in, int[] b) {
			this(a, in);
			if(n != b.length)
				throw new DimensionMismatchException(n, b.length);
			
			for(int i = 0; i < n; i++)
				vec[i] = eval(a[i], b[i]);
		}
	}
	
	
	
	
	
	
	// =============== DIM CHECKS ==============
	final private static void dimAssess(final int a) { if(a < MIN_ACCEPTABLE_VEC_LEN) throw new IllegalArgumentException(VEC_LEN_ERR + a); }
	final static public void checkDims(final boolean[] a) 	{ dimAssess(a.length); }
	final static public void checkDims(final int[] a) 		{ dimAssess(a.length); }
	final static public void checkDims(final double[] a) 	{ dimAssess(a.length); }
	
	final private static void dimAssessPermitEmpty(final int a) 	{ if(a < 0) throw new IllegalArgumentException(VEC_LEN_ERR + a); }
	final static public void checkDimsPermitEmpty(final boolean[] a){ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final int[] a) 	{ dimAssessPermitEmpty(a.length); }
	final static public void checkDimsPermitEmpty(final double[] a) { dimAssessPermitEmpty(a.length); }
	
	final private static void dimAssess(final int a, final int b) { if(a != b) throw new DimensionMismatchException(a, b); dimAssess(a); }
	final static public void checkDims(final boolean[] a, final boolean[] b){ dimAssess(a.length, b.length); }
	final static public void checkDims(final int[] a, final int[] b) 		{ dimAssess(a.length, b.length); }
	final static public void checkDims(final double[] a, final double[] b) 	{ dimAssess(a.length, b.length); }
	
	final private static void dimAssessPermitEmpty(final int a, final int b) 			{ if(a != b) throw new DimensionMismatchException(a, b); dimAssessPermitEmpty(a); }
	final static public void checkDimsPermitEmpty(final boolean[] a, final boolean[] b)	{ dimAssessPermitEmpty(a.length, b.length); }
	final static public void checkDimsPermitEmpty(final int[] a, final int[] b) 		{ dimAssessPermitEmpty(a.length, b.length); }
	final static public void checkDimsPermitEmpty(final double[] a, final double[] b)	{ dimAssessPermitEmpty(a.length, b.length); }

	
	
	
	
	
	// ====================== MATH FUNCTIONS =======================
	/**
	 * Calculate the absolute value of the values in the vector and return a copy.
	 * Depending on {@link GlobalState} parallelism settings, auto schedules parallel
	 * or serial job.
	 * @param a
	 * @return absolute value of the vector
	 */
	public static double[] abs(final double[] a) {
		checkDimsPermitEmpty(a);
		final double[] b= new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.abs(a[i]);
		return b;
	}
	
	
	
	
	/**
	 * Add two vectors.
	 * Depending on {@link GlobalState} parallelism settings, auto schedules parallel
	 * or serial job.
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if dims do not match
	 * @return the result of adding two vectors
	 */
	public static double[] add(final double[] a, final double[] b) {
		checkDimsPermitEmpty(a, b);
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] + b[i];
		
		return ab;
	}
	
	
	
	
	
	// ================= arange ==================
	private static int check_arange_return_len(int st, int en, int in) {
		if(in == 0) throw new IllegalArgumentException("increment cannot equal zero");
		if(st > en && in > 0) throw new IllegalArgumentException("increment can't be positive for this range");
		if(st < en && in < 0) throw new IllegalArgumentException("increment can't be negative for this range");
		
		int length = FastMath.abs(en - st);
		if(length == 0) throw new IllegalArgumentException("start_inc ("+st+") cannot equal end_exc ("+en+")");
		if(length%FastMath.abs(in)!=0) throw new IllegalArgumentException("increment will not create evenly spaced elements");
		if(length > GlobalState.MAX_ARRAY_SIZE) throw new IllegalArgumentException("array would be too long");
		
		return length;
	}
	
	/**
	 * Create a range of values starting at zero (inclusive)
	 * and continuing to the provided length (exclusive). 
	 * <br>EX: <tt>arange(10) = {0,1,2,3,4,5,6,7,8,9}</tt>
	 * @param length
	 * @throws IllegalArgumentException if the length exceeds 
	 * {@value GlobalState#MAX_ARRAY_SIZE} or if length == 0
	 * @return a range of values
	 */
	public static int[] arange(final int length) {
		return arange(0, length, 1);
	}
	
	/**
	 * Create a range of values starting at <tt>start_inc</tt> (inclusive) and 
	 * continuing to <tt>end_exc</tt> (exclusive). 
	 * <br>EX 1: <tt>arange(2,5) = {2,3,4}</tt>
	 * <br>EX 2: <tt>arange(5,2) = {5,4,3}</tt>
	 * @param start_inc - the beginning index, inclusive
	 * @param end_exc - the stopping index, exclusive
	 * @throws IllegalArgumentException if start_inc == end_exc or if the difference
	 * between start and end exceeds {@value GlobalState#MAX_ARRAY_SIZE}
	 * @return a range of values
	 */
	public static int[] arange(final int start_inc, final int end_exc) {
		return arange(start_inc, end_exc, start_inc>end_exc?-1:1);
	}
	
	/**
	 * Create a range of values starting at <tt>start_inc</tt> (inclusive) and 
	 * continuing to <tt>end_exc</tt> (exclusive). 
	 * <br>EX 1: <tt>arange(2, 5, 1) = {2,3,4}</tt>
	 * <br>EX 2: <tt>arange(5, 2,-1) = {5,4,3}</tt>
	 * <br>EX 3: <tt>arange(0,10, 2) = {0,2,4,6,8}</tt>
	 * @param start_inc - the beginning index, inclusive
	 * @param end_exc - the stopping index, exclusive
	 * @param increment - the amount to space values by
	 * @throws IllegalArgumentException if start_inc == end_exc, if the difference
	 * between start and end exceeds {@value GlobalState#MAX_ARRAY_SIZE}, if the values
	 * cannot be evenly distributed given the increment value
	 * @return a range of values
	 */
	public static int[] arange(final int start_inc, final int end_exc, final int increment) {
		int length = check_arange_return_len(start_inc, end_exc, increment) / FastMath.abs(increment);
		
		int i, j;
		final int[] out = new int[length];
		if(increment < 0)
			for(i = start_inc, j = 0; i > end_exc; i+=increment, j++) out[j] = i;
		else
			for(i = start_inc, j = 0; i < end_exc; i+=increment, j++) out[j] = i;
		
		return out;
	}
	
	
	
	/**
	 * Return the index of the max element in the vector. In the case
	 * of a tie, the first "max" element ordinally will be returned.
	 * @param v
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the idx of the max element
	 */
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
	
	
	/**
	 * Return the index of the min element in the vector. In the case
	 * of a tie, the first "min" element ordinally will be returned.
	 * @param v
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the idx of the min element
	 */
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
	
	/**
	 * Given a vector, returns a vector of ints corresponding to the position
	 * of the original elements the indices in which they would be ordered were they sorted.
	 * <br> EX: <tt>argSort({5,1,3,4}) = {1,2,3,0}</tt>, where reordering the input vector
	 * in the index order <tt>{1,2,3,0}</tt> would effectively sort the input vector.
	 * @param a
	 * @throws IllegalArgumentException if the input vector is empty
	 * @return the ascending sort order of indices
	 */
	public static int[] argSort(final double[] a) {
		checkDims(a);
		return ArgSorter.argsort(a);
	}
	
	/**
	 * Given a vector, returns a vector of ints corresponding to the position
	 * of the original elements the indices in which they would be ordered were they sorted.
	 * <br> EX: <tt>argSort({5,1,3,4}) = {1,2,3,0}</tt>, where reordering the input vector
	 * in the index order <tt>{1,2,3,0}</tt> would effectively sort the input vector.
	 * @param a
	 * @throws IllegalArgumentException if the input vector is empty
	 * @return the ascending sort order of indices
	 */
	public static int[] argSort(final int[] a) {
		checkDims(a);
		return ArgSorter.argsort(a);
	}
	
	/**
	 * Class to arg sort double and int arrays
	 * @author Taylor G Smith
	 */
	abstract static class ArgSorter {
		static int[] argsort(final double[] a) {
	        return argsort(a, true);
	    }
		
		private static Integer[] _arange(int len) {
			Integer[] range = new Integer[len];
	        for (int i = 0; i < len; i++)
	            range[i] = i;
	        
	        return range;
		}

	    static int[] argsort(final double[] a, final boolean ascending) {
	        Integer[] indexes = _arange(a.length);
	        
	        Arrays.sort(indexes, new Comparator<Integer>() {
	            @Override
	            public int compare(final Integer i1, final Integer i2) {
	                return (ascending ? 1 : -1) * Double.compare(a[i1], a[i2]);
	            }
	        });
	        
	        return asArray(indexes);
	    }
		
		static int[] argsort(final int[] a) {
			return argsort(a, true);
		}
	    
	    static int[] argsort(final int[] a, final boolean ascending) {
	        Integer[] indexes = _arange(a.length);
	        
	        Arrays.sort(indexes, new Comparator<Integer>() {
	            @Override
	            public int compare(final Integer i1, final Integer i2) {
	                return (ascending ? 1 : -1) * Integer.compare(a[i1], a[i2]);
	            }
	        });
	        
	        return asArray(indexes);
	    }

	    @SafeVarargs
		static <T extends Number> int[] asArray(final T... a) {
	        int[] b = new int[a.length];
	        for (int i = 0; i < b.length; i++) {
	            b[i] = a[i].intValue();
	        }
	        
	        return b;
	    }
	}
	
	/**
	 * Coerce an int vector to a double vector. If the
	 * input vector, will return an empty double vector
	 * @param a
	 * @return the double vector
	 */
	public static double[] asDouble(final int[] a) {
		checkDimsPermitEmpty(a);
		final int n = a.length;
		final double[] d = new double[n];
		
		for(int i = 0; i < n; i++)
			d[i] = (double)a[i];
		
		return d;
	}
	
	/**
	 * Concatenate two vectors together.
	 * <br>EX: cat({1,2,3}, {4,5,6}) = {1,2,3,4,5,6}
	 * @param a
	 * @param b
	 * @return the concatenation of A and B
	 */
	final public static int[] cat(final int[] a, final int[] b) {
		checkDimsPermitEmpty(a);
		checkDimsPermitEmpty(b);
		
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
	
	/**
	 * Concatenate two vectors together.
	 * <br>EX: cat({1,2,3}, {4,5,6}) = {1,2,3,4,5,6}
	 * @param a
	 * @param b
	 * @return the concatenation of A and B
	 */
	final public static double[] cat(final double[] a, final double[] b) {
		checkDimsPermitEmpty(a);
		checkDimsPermitEmpty(b);
		
		final int na = a.length, nb = b.length, n = na+nb;
		if(na == 0) return copy(b);
		if(nb == 0) return copy(a);
		
		final double[] res = new double[n];
		for(int i = 0; i < na; i++)
			res[i] = a[i];
		for(int i = 0; i < nb; i++)
			res[i+na] = b[i];
		
		return res;
	}
	
	/**
	 * Zero-center a vector around the mean
	 * @param a
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the centered vector
	 */
	final public static double[] center(final double[] a) {
		return center(a, mean(a));
	}
	
	/**
	 * Zero-center a vector around a value
	 * @param a
	 * @param value
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the centered vector
	 */
	final public static double[] center(final double[] a, final double value) {
		checkDims(a);
		
		final double[] copy = new double[a.length];
		System.arraycopy(a, 0, copy, 0, a.length);
		for(int i = 0; i < a.length; i++)
			copy[i] = a[i] - value;
		return copy;
	}
	
	/**
	 * Get all the complete (non-NaN) values in a vector
	 * @param d
	 * @return the complete vector
	 */
	public static double[] completeCases(final double[] d) {
		checkDimsPermitEmpty(d);
		
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
		for(double b: a)
			if(Double.isNaN(b))
				return true;
		
		return false;
	}
	
	/**
	 * Return a copy of a boolean array
	 * @param b
	 * @return the copy
	 */
	public static boolean[] copy(final boolean[] b) {
		if(null == b)
			return null;
		
		final boolean[] copy = new boolean[b.length];
		System.arraycopy(b, 0, copy, 0, b.length);
		return copy;
	}
	
	/**
	 * Return a copy of an int array
	 * @param i
	 * @return the copy
	 */
	public static int[] copy(final int[] i) {
		if(null == i)
			return null;
			
		final int[] copy = new int[i.length];
		System.arraycopy(i, 0, copy, 0, i.length);
		return copy;
	}
	
	/**
	 * Return a copy of a double array
	 * @param d
	 * @return the copy
	 */
	public static double[] copy(final double[] d) {
		if(null == d)
			return null;
		
		final double[] copy = new double[d.length];
		System.arraycopy(d, 0, copy, 0, d.length);
		return copy;
	}
	
	/**
	 * Return a copy of a String array
	 * @param s
	 * @return the copy
	 */
	public static String[] copy(final String[] s) {
		if(null == s)
			return null;
		
		final String[] copy = new String[s.length];
		System.arraycopy(s, 0, copy, 0, s.length);
		return copy;
	}
	
	/**
	 * Returns a shallow copy of the arg ArrayList. If the generic
	 * type is immutable (an instance of Number, String, etc) will
	 * act as a deep copy.
	 * @param a
	 * @throws NullPointerException if arg is null
	 * @return a shallow copy
	 */
	public static <T> ArrayList<T> copy(final ArrayList<T> a) {
		final ArrayList<T> copy = new ArrayList<T>(a.size());
		
		for(T i: a)
			copy.add(i);
		
		return copy;
	}
	
	/**
	 * Computes the cosine similarity between two vectors.
	 * @param a
	 * @param b
	 * @throws IllegalArgumentException if either a or b is empty
	 * @throws DimensionMismatchException if dims don't match
	 * @return the cosine similarity
	 */
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
	
	public static double[] cumsum(final double[] a) {
		checkDimsPermitEmpty(a);
		
		final int n = a.length;
		if(n == 0)
			return new double[]{};
		
		double[] b = new double[n];
		double sum = 0;
		for(int i = 0; i < n; i++) {
			sum += a[i];
			b[i] = sum;
		}
		
		return b;
	}
	
	/**
	 * Divide one vector by another
	 * @param numer
	 * @param by
	 * @throws DimensionMismatchException if the dims don't match
	 * @return the quotient vector
	 */
	public static double[] divide(final double[] numer, final double[] by) {
		checkDimsPermitEmpty(numer, by);
		
		final double[] ab = new double[numer.length];
		for(int i = 0; i < numer.length; i++)
			ab[i] = numer[i] / by[i];
		
		return ab;
	}
	
	/**
	 * Returns true if every element in the vector A
	 * exactly equals the corresponding element in the vector B
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if all equal, false otherwise
	 */
	public static boolean equalsExactly(final int[] a, final int[] b) {
		if(null == a && null == b)
			return true;
		if(null == a ^ null == b)
			return false;
		if(a.length != b.length)
			return false;
		
		for(int i = 0; i < a.length; i++)
			if(a[i] != b[i])
				return false;
		
		return true;
	}
	
	/**
	 * Returns true if every element in the vector A
	 * exactly equals the corresponding element in the vector B
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if all equal, false otherwise
	 */
	public static boolean equalsExactly(final boolean[] a, final boolean[] b) {
		if(null == a && null == b)
			return true;
		if(null == a ^ null == b)
			return false;
		if(a.length != b.length)
			return false;
		
		for(int i = 0; i < a.length; i++)
			if(a[i] != b[i])
				return false;
		return true;
	}
	
	/**
	 * Returns true if every element in the vector A
	 * exactly equals the corresponding element in the vector B
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if all equal, false otherwise
	 */
	public static boolean equalsExactly(final String[] a, final String[] b) {
		if(null == a && null == b)
			return true;
		if(null == a ^ null == b)
			return false;
		if(a.length != b.length)
			return false;
		
		for(int i = 0; i < a.length; i++)
			if(!a[i].equals(b[i]))
				return false;
		return true;
	}
	
	/**
	 * Returns true if every element in the vector A
	 * exactly equals the corresponding element in the vector B.
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if all equal, false otherwise
	 */
	public static boolean equalsExactly(final double[] a, final double[] b) {
		return equalsWithTolerance(a, b, 0);
	}
	
	
	/**
	 * Returns true if every element in the vector A
	 * equals the corresponding element in the vector B within
	 * a default tolerance of {@link Precision#EPSILON}
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if all equal, false otherwise
	 */
	public static boolean equalsWithTolerance(final double[] a, final double[] b) {
		return equalsWithTolerance(a, b, Precision.EPSILON);
	}
	
	
	/**
	 * Returns true if every element in the vector A
	 * equals the corresponding element in the vector B within
	 * a provided tolerance
	 * @param a
	 * @param b
	 * @param eps
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if all equal, false otherwise
	 */
	public static boolean equalsWithTolerance(final double[] a, final double[] b, final double eps) {
		if(null == a && null == b)
			return true;
		if(null == a ^ null == b)
			return false;
		if(a.length != b.length)
			return false;
		
		for(int i = 0; i < a.length; i++) {
			if(Double.isNaN(a[i]) && Double.isNaN(b[i]))
				continue;
			
			if( !Precision.equals(a[i], b[i], eps) )
				return false;
		}
		
		return true;
	}
	
	/**
	 * Apply the {@link FastMath#exp(double)} function
	 * across a vector.
	 * @param a
	 * @return a vector of corresponding exp'd values
	 */
	public static double[] exp(final double[] a) {
		checkDimsPermitEmpty(a);
		
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
		checkDimsPermitEmpty(a);
		
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
	 * @throws DimensionMismatchException if the dims don't match
	 * @return the inner product between a and b
	 */
	public static double innerProduct(final double[] a, final double[] b) {
		checkDimsPermitEmpty(a, b);
		double sum = 0.0;
		for(int i = 0; i < a.length; i++)
			sum += a[i] * b[i];
		
		return sum;
	}
	
	/**
	 * Compute the interquartile range in a vector
	 * @param a
	 * @throws IllegalArgumentException if the input vector is empty
	 * @return the interquartile range
	 */
	public static double iqr(final double[] a) {
		checkDims(a);
		DescriptiveStatistics d = new DescriptiveStatistics(a);
		return d.getPercentile(75) - d.getPercentile(25);
	}
	
	/**
	 * Assess whether two vectors are orthogonal, i.e., their inner product is 0.
	 * @param a
	 * @param b
	 * @throws DimensionMismatchException if the dims don't match
	 * @return true if the inner product equals 0
	 */
	public static boolean isOrthogonalTo(final double[] a, final double[] b) {
		// Will auto determine whether parallel is necessary or allowed...
		return Precision.equals(innerProduct(a, b), 0, Precision.EPSILON);
	}
	
	/**
	 * Computes the <tt>L<sub>1</sub></tt> norm, or the sum
	 * of the absolute values in the vector.
	 * @param a
	 * @return the norm
	 */
	public static double l1Norm(final double[] a) {
		return sum(abs(a));
	}
	
	/**
	 * Compute the <tt>L<sub>2</sub></tt> (Euclidean) norm, or the sqrt
	 * of the sum of squared terms in the vector
	 * @param a
	 * @return the norm
	 */
	public static double l2Norm(final double[] a) {
		return FastMath.sqrt(innerProduct(a, a));
	}
	
	/**
	 * Calculate the log of the vector.
	 * @param a
	 * @return the log of the vector
	 */
	public static double[] log(final double[] a) {
		checkDimsPermitEmpty(a);
		final double[] b = new double[a.length];
		for(int i = 0; i < a.length; i++)
			b[i] = FastMath.log(a[i]);
		return b;
	}
	
	/**
	 * Return the <tt>L<sub>P</sub></tt> or Minkowski norm
	 * @param a
	 * @param p
	 * @return the <tt>L<sub>P</sub></tt> norm
	 */
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
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the max in the vector
	 */
	final public static double max(final double[] a) {
		checkDims(a);
		
		double max = SIGNED_MIN;
		for(double d : a)
			if(d > max)
				max = d;
		return max;
	}
	
	/**
	 * Calculate the mean of the vector, NaN if it's empty
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
	 * @throws IllegalArgumentException if the vector is empty
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
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the min in the vector
	 */
	final public static double min(final double[] a) {
		checkDims(a);
		
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
	 * @throws DimensionMismatchException if the vector dims don't match
	 * @return the product of two vectors
	 */
	public static double[] multiply(final double[] a, final double[] b) {
		checkDimsPermitEmpty(a, b);
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
		int ct = 0;
		for(double d: a)
			if(Double.isNaN(d))
				ct++;
		
		return ct;
	}
	
	
	/**
	 * Identify the max value in the vector, excluding NaNs
	 * @param a
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the max in the vector excluding NaNs; returns NaN if completely NaN vector
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
	 * @throws IllegalArgumentException if the vector is empty
	 * @return the min in the vector excluding NaNs; returns NaN if completely NaN vector
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
	 * @return the mean in the vector excluding NaNs; returns NaN if completely NaN vector
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
	 * @return the median in the vector excluding NaNs; returns NaN if completely NaN vector
	 */
	public static double nanMedian(final double[] a) {
		// handles case of whether vector length 1
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
	 * Multiplies the entire vector by -1
	 * @param a
	 * @return the negative of the vector
	 */
	public static double[] negative(final double[] a) {
		return scalarMultiply(a, -1);
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
	 * Shuffle in the input
	 * @param in
	 * @return a shuffled int array
	 */
	public static int[] permutation(final int[] in) {
		return permutation(in, GlobalState.DEFAULT_RANDOM_STATE);
	}
	
	/**
	 * Shuffle in the input
	 * @param in
	 * @param rand - a random seed
	 * @return a shuffled int array
	 */
	public static int[] permutation(final int[] in, final Random rand) {
		checkDimsPermitEmpty(in);

		final int m = in.length;
		ArrayList<Integer> recordIndices = new ArrayList<Integer>(m);
		
		for(int i = 0; i < m; i++) 
			recordIndices.add(i);
		
		Collections.shuffle(recordIndices, rand);
		final int[] out = new int[m];
		for(int i = 0; i < m; i++)
			out[i] = recordIndices.get(i);
		
		return out;
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
		checkDimsPermitEmpty(a);
		
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
		checkDimsPermitEmpty(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] + b;
		
		return ab;
	}
	
	public static double[] scalarDivide(final double[] a, final double b) {
		checkDimsPermitEmpty(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] / b;
		
		return ab;
	}
	
	public static double[] scalarMultiply(final double[] a, final double b) {
		checkDimsPermitEmpty(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] * b;
		
		return ab;
	}
	
	public static double[] scalarSubtract(final double[] a, final double b) {
		checkDimsPermitEmpty(a);
		
		final double[] ab = new double[a.length];
		for(int i = 0; i < a.length; i++)
			ab[i] = a[i] - b;
		
		return ab;
	}
	
	public static double[] slice(final double[] a, final int startInc, final int endExc) {
		checkDims(a);
		
		if(endExc > a.length)
			throw new ArrayIndexOutOfBoundsException(endExc);
		if(startInc < 0 || startInc > a.length)
			throw new ArrayIndexOutOfBoundsException(startInc);
		if(startInc > endExc)
			throw new IllegalArgumentException("start index cannot exceed end index");
		if(startInc == endExc)
			return new double[]{};
		
		final double[] out = new double[endExc - startInc];
		for(int i = startInc, j = 0; i < endExc; i++, j++)
			out[j] = a[i];
		
		return out;
	}
	
	public static int[] slice(final int[] a, final int startInc, final int endExc) {
		checkDims(a);
		
		if(endExc > a.length)
			throw new ArrayIndexOutOfBoundsException(endExc);
		if(startInc < 0 || startInc > a.length)
			throw new ArrayIndexOutOfBoundsException(startInc);
		if(startInc > endExc)
			throw new IllegalArgumentException("start index cannot exceed end index");
		if(startInc == endExc)
			return new int[]{};
		
		final int[] out = new int[endExc - startInc];
		for(int i = startInc, j = 0; i < endExc; i++, j++)
			out[j] = a[i];
		
		return out;
	}
	
	public static double[] sortAsc(final double[] a) {
		checkDimsPermitEmpty(a);
		
		final int n = a.length;
		if(n == 0)
			return new double[]{};
		
		final double[] b = copy(a);
		Arrays.sort(b);
		return b;
	}
	
	public static int[] sortAsc(final int[] a) {
		checkDimsPermitEmpty(a);
		
		final int n = a.length;
		if(n == 0)
			return new int[]{};
		
		final int[] b = copy(a);
		Arrays.sort(b);
		return b;
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
		checkDimsPermitEmpty(from, subtractor);
		
		final double[] ab = new double[from.length];
		for(int i = 0; i < from.length; i++)
			ab[i] = from[i] - subtractor[i];
		
		return ab;
	}
	
	public static double sum(final double[] a) {
		double sum = 0.0;
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
	
	public static LinkedHashSet<Double> unique(final double[] arr) {
		final LinkedHashSet<Double> out = new LinkedHashSet<>();
		for(Double t: arr)
			out.add(t);
		return out;
	}
	
	public static LinkedHashSet<Integer> unique(final int[] arr) {
		final LinkedHashSet<Integer> out = new LinkedHashSet<>();
		for(Integer t: arr)
			out.add(t);
		return out;
	}
	
	final public static double var(final double[] a) {
		return var(a, DEF_SUBTRACT_ONE_VAR);
	}
	
	/**
	 * Compute the variance of a vector given the mean.
	 * @param a
	 * @param mean
	 * @return
	 */
	final public static double var(final double[] a, final double mean) {
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
		
		return sum / ((double)a.length - (n_minus_one ? 1.0 : 0.0));
	}
	
	public static double[][] vstack(final double[] a, final double[] b) {
		checkDimsPermitEmpty(a,b);
		
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
	
	public static double[] where(final DoubleSeries series, final double[] x, final double[] y) {
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
	
	/*
	public static double[] where(final VecDoubleSeries series, final double x, final double[] y) {
		return where(series, rep(x, series.get().length), y);
	}
	
	public static double[] where(final VecDoubleSeries series, final double[] x, final double y) {
		return where(series, x, rep(y, series.get().length));
	}
	
	public static double[] where(final VecDoubleSeries series, final double x, final double y) {
		return where(series, rep(x,series.get().length), rep(y,series.get().length));
	}
	*/
}
