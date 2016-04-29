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

package com.clust4j.algo.preprocess;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.ParallelChunkingTask;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class BoxCoxTransformer extends Transformer {
	private static final long serialVersionUID = -5397818601304593058L;
	public static final double DEF_LAM_MIN = 0.05;
	public static final double DEF_LAM_MAX = 0.5;
	public static final double DEF_LAM_INC = 0.05;
	static final double zero = 1e-12;
	
	/*
	 * Lambda search parameters
	 */
	final private double lambda_min;
	final private double lambda_max;
	final private double increment;
	
	volatile protected double[] lambdas;
	volatile protected double[] shift;
	
	
	private BoxCoxTransformer(BoxCoxTransformer bc) {
		this.lambdas = VecUtils.copy(bc.lambdas);
		this.shift = VecUtils.copy(bc.shift);
		this.lambda_min = bc.lambda_min;
		this.lambda_max = bc.lambda_max;
		this.increment = bc.increment;
	}
	
	public BoxCoxTransformer() {
		this(DEF_LAM_MIN, DEF_LAM_MAX, DEF_LAM_INC);
	}
	
	public BoxCoxTransformer(double lam_min, double lam_max, double increment) {
		ensureAllPositive(lam_min, lam_max, increment);
		
		if(lam_max <= lam_min)
			throw new IllegalArgumentException("lam_max must exceed lam_min");
		
		this.lambda_min = lam_min;
		this.lambda_max = lam_max;
		this.increment = increment;
	}
	
	
	private static void ensureAllPositive(double...a) {
		for(double d: a) {
			if(d <= 0.0) {
				throw new IllegalArgumentException("all values must be positive");
			}
		}
	}
	
	
	@Override
	protected void checkFit() {
		if(null == lambdas)
			throw new ModelNotFitException("model not yet fit");
	}

	/**
	 * Inverse transform your matrix. Note: this suffers some
	 * accuracy issues due to the log base
	 */
	@Override
	public RealMatrix inverseTransform(RealMatrix X) {
		checkFit();
		
		final int m = X.getRowDimension();
		final int n = X.getColumnDimension();
		
		if(n != shift.length)
			throw new DimensionMismatchException(n, shift.length);
		
		double[][] x = X.getData();
		for(int j = 0; j < n; j++) {
			double lam = lambdas[j];
			double ool = 1.0 / lam;
			
			for(int i = 0; i < m; i++) {
				// If the lambda is near zero, exp to reverse the log:
				if(lam < zero) {
					x[i][j] = FastMath.exp(x[i][j]);
				} else {
					x[i][j] *= lam;
					x[i][j] += 1;
					x[i][j] = FastMath.pow(x[i][j], ool);
				}
				
				// Add back the shift value:
				x[i][j] += shift[j];
			}
		}
		
		// Implicit copy in the getData()
		return new Array2DRowRealMatrix(x, false);
	}

	@Override
	public BoxCoxTransformer copy() {
		return new BoxCoxTransformer(this);
	}
	
	
	/**
	 * This class estimates lambda values for each row on a transposed matrix,
	 * X, and performs goodness-of-fit tests on each set of estimates.
	 * @author Taylor G Smith
	 */
	static class ParallelLambdaEstimator extends ParallelChunkingTask<double[]> {
		private static final long serialVersionUID = 6510959845256491305L;
		
		private double[] shift;
		private double[] lambdas;
		private int lo, hi;
		private double lmin, lmax, increment;

		public ParallelLambdaEstimator(double[][] X, double[] shift, double lmin, double lmax, double inc) {
			super(X);
			
			// Init lambdas and shift
			this.shift = shift;
			this.lambdas = new double[shift.length];
			this.lmin = lmin;
			this.lmax = lmax;
			this.increment = inc;
			
			this.lo = 0;
			this.hi = strategy.getNumChunks(X);
		}
		
		public ParallelLambdaEstimator(ParallelLambdaEstimator instance, int lo, int hi) {
			super(instance);
			
			this.shift = instance.shift;
			this.lambdas = instance.lambdas;
			this.lmin = instance.lmin;
			this.lmax = instance.lmax;
			this.increment = instance.increment;
			
			this.lo = lo;
			this.hi = hi;
		}

		@Override
		public double[] reduce(Chunk chunk) {
			double[][] x = chunk.get();
			int start = chunk.start; // retrieve idx of shift & lambda
			
			NormalDistribution norm;
			double shift_factor;
			for(double[] feature: x) {
				int m = feature.length;
				shift_factor = this.shift[start];
				
				// for each lambda in the range:
				double min_kolmogorov = Double.POSITIVE_INFINITY;
				double best_lambda = Double.NaN;
				for(double lambda = lmin; lambda <= lmax; lambda += increment) {
					
					// We need to keep track of the mean as we go... also stddev
					double sum = 0.0, sumSq = 0.0, mean, var, std;
					
					// Create the transformed feature:
					double[] trans = new double[m];
					for(int j = 0; j < m; j++) {
						trans[j] = lambdaTransform(feature[j], shift_factor, lambda);
						sumSq += (trans[j] * trans[j]);
						sum += trans[j];
					}
					
					// Update mean & stddev:
					mean = sum / (double) m;
					var = (sumSq - (sum*sum)/(double)m ) / ((double)m - 1.0);
					std = FastMath.sqrt(var);
					norm = new NormalDistribution(mean, std);
					
					// Test goodness of fit
					double ks = goodnessOfFitTest(trans, norm);
					if(ks < min_kolmogorov) {
						min_kolmogorov = ks;
						best_lambda = lambda;
					}
				}
				
				this.lambdas[start] = best_lambda;
				start++;
			}
			
			// Since this works in place, this is unnecessary,
			// but we have to match the signature of the API
			return lambdas;
		}
		
		/**
		 * Perform test for normality using the Kolmogorov-Smirnov test
		 * @param transformed
		 * @return
		 */
		static double goodnessOfFitTest(double[] transformed, NormalDistribution norm) {
			return new KolmogorovSmirnovTest().kolmogorovSmirnovTest(norm, transformed);
		}

		@Override
		protected double[] compute() {
			if(hi - lo <= 1) { // generally should equal one...
				return reduce(chunks.get(lo));
			} else {
				int mid = this.lo + (this.hi - this.lo) / 2;
				ParallelLambdaEstimator left = new ParallelLambdaEstimator(this, lo, mid);
				ParallelLambdaEstimator right= new ParallelLambdaEstimator(this, mid, hi);
				
				// These ops happen in place
				left.fork();
				right.compute();
				left.join();
				
				return this.lambdas;
			}
		}
		
		static double[] doAll(double[][] X, double[] shift, double min, double max, double inc) {
			return getThreadPool().invoke(new ParallelLambdaEstimator(X, shift, min, max, inc));
		}
	}
	
	
	
	/**
	 * Shift and transform the feature
	 * @param y
	 * @param shift
	 * @param lambda
	 * @return
	 */
	static double lambdaTransform(double y, double shift, double lambda) {
		double shifted = FastMath.max(y + shift, 1.0);
		
		//if(shifted < 1.0) {
			/*
			 * We shift everything up to 1.0, so if it's less
			 * than 1.0, we know this val is less than the smallest
			 * num we saw in the training vector.
			 */
		//	throw new NotStrictlyPositiveException(shifted);
		//} else 
			
		if(FastMath.abs(lambda) < zero) {
			return FastMath.log(shifted);
		} else {
			return (FastMath.pow(shifted, lambda) - 1.0) / lambda;
		}
	}

	@Override
	public BoxCoxTransformer fit(RealMatrix X) {
		synchronized(fitLock) {
			final int n = X.getColumnDimension(), m = X.getRowDimension();
			
			// If m < 2, we can't effectively measure std-dev and thus can't estimate
			if(m < 2) {
				throw new IllegalArgumentException("need at least two observations");
			}
			
			this.shift = new double[n];
			
			// Transpose so we can use VecUtils more efficiently,
			// and then chunk the data for parallel operation
			double[][] x = X.transpose().getData();
			for(int j = 0; j < n; j++) {
				double fac = 0.0;
				double min = VecUtils.min(x[j]);
				if(min < 0) {
					fac = 1.0 - min;
				}
				
				this.shift[j] = fac;
			}
			
			
			// Estimate the lambdas in parallel...
			try {
				this.lambdas = ParallelLambdaEstimator.doAll(x, shift, lambda_min, lambda_max, increment);
			} catch(NotStrictlyPositiveException nspe) {
				throw new IllegalArgumentException("is one of your columns a constant?", nspe);
			}
			
			return this;
		}
	}

	@Override
	public RealMatrix transform(RealMatrix data) {
		return new Array2DRowRealMatrix(transform(data.getData()), false);
	}

	@Override
	public double[][] transform(double[][] data) {
		checkFit();
		MatUtils.checkDimsForUniformity(data);
		
		final int m = data.length;
		final int n = data[0].length;
		
		if(n != shift.length)
			throw new DimensionMismatchException(n, shift.length);
		
		double[][] X =  new double[m][n];
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = lambdaTransform(data[i][j], shift[j], lambdas[j]);
			}
		}
		
		return X;
	}
}
