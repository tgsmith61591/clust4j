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

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class StandardScaler extends Transformer {
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
	public StandardScaler fit(RealMatrix data) {
		synchronized(fitLock) {
			final int m = data.getRowDimension();
			final int n = data.getColumnDimension();
			
			if(m < 2)
				throw new IllegalArgumentException("cannot "
					+ "meaningfully compute standard deviation "
					+ "on fewer than two observations");

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
	public RealMatrix transform(RealMatrix data) {
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
	
	@Override
	public RealMatrix inverseTransform(RealMatrix X) {
		checkFit();
		
		// This effectively copies, so no need to do a copy later
		double[][] data = X.getData();
		final int m = data.length;
		final int n = data[0].length;
		
		if(n != means.length)
			throw new DimensionMismatchException(n, means.length);
		
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				data[i][j] *= stdevs[j]; // first re-scale
				data[i][j] += means[j];  // then add means
			}
		}
		
		return new Array2DRowRealMatrix(data, false);
	}
}
