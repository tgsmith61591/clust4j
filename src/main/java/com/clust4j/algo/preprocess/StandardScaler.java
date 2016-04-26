package com.clust4j.algo.preprocess;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class StandardScaler extends PreProcessor {
	private static final long serialVersionUID = 8999017379613060521L;
	volatile double[] means;
	volatile double[] stdevs;
	
	private StandardScaler(StandardScaler instance) {
		this.means = VecUtils.copy(instance.means);
		this.stdevs= VecUtils.copy(instance.stdevs);
	}
	
	public StandardScaler() {
	}
	
	
	@Override
	protected void checkFit() {
		if(null == means)
			throw new ModelNotFitException("model not yet fit");
	}
	
	@Override
	public StandardScaler copy() {
		return new StandardScaler(this);
	}

	@Override
	public StandardScaler fit(AbstractRealMatrix data) {
		synchronized(fitLock) {
			final int m = data.getRowDimension();
			final int n = data.getColumnDimension();

			// need to mean center...
			this.means = new double[n];
			this.stdevs= new double[n];
			
			final double[][] X = data.getData();
			
			for(int col = 0; col < n; col++) {
				double var, std, mn;
				double sumSq = 0.0;
				double sum   = 0.0;
				
				for(int row = 0; row < m; row++) {
					sumSq += X[row][col] * X[row][col];
					sum += X[row][col];
				}
				
				/*
				 * A naive algorithm to calculate the estimated variance (1M):
				 * 
				 * Let n = 0, Sum = 0, SumSq = 0 
				 * For each datum x: 
				 *   n = n + 1 
				 *   Sum = Sum + x 
				 *   SumSq = SumSq + x * x 
				 * Var = (SumSq - (Sum * Sum) / n) / (n - 1)
				 * 
				 * @see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
				 */
				var = (sumSq - (sum*sum)/(double)m ) / ((double)m - 1.0);
				std = m < 2 ? Double.NaN : FastMath.sqrt(var);
				mn  = sum / (double)m;
				
				means[col] = mn;
				stdevs[col]= std;
			}
			
			return this;
		}
	}

	@Override
	public AbstractRealMatrix transform(AbstractRealMatrix data) {
		return new Array2DRowRealMatrix(transform(data.getData()), false);
	}

	@Override
	public double[][] transform(double[][] data) {
		checkFit();
		MatUtils.checkDimsForUniformity(data);
		
		final int m = data.length;
		final int n = data[0].length;
		
		if(n != means.length)
			throw new DimensionMismatchException(n, means.length);

		double[][] X = new double[m][n];
		// second pass, subtract to center:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = (data[i][j] - means[j]) / stdevs[j];
			}
		}
		
		// assign
		return X;
	}
}
