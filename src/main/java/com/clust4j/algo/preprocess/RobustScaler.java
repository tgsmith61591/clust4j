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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class RobustScaler extends Transformer {
	private static final long serialVersionUID = 9139185680482876266L;
	volatile private MedianCenterer centerer;
	volatile double[] scale;
	
	
	
	private RobustScaler(RobustScaler rs) {
		this.centerer = null == rs.centerer ? null : rs.centerer.copy();
		this.scale = VecUtils.copy(rs.scale);
	}
	
	public RobustScaler() {
	}
	
	

	@Override
	protected void checkFit() {
		if(null == centerer)
			throw new ModelNotFitException("model not yet fit");
	}

	@Override
	public RealMatrix inverseTransform(RealMatrix X) {
		checkFit();
		
		// This effectively copies, so no need to do a copy later
		double[][] data = X.getData();
		final int m = data.length;
		final int n = data[0].length;
		
		if(n != this.centerer.medians.length)
			throw new DimensionMismatchException(n, this.centerer.medians.length);
		
		// First, multiply back by scales
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				data[i][j] *= scale[j];
				
				// To avoid a second pass of O(M*N), we
				// won't call the inverseTransform in the centerer,
				// we will just explicitly add the median back here.
				data[i][j] += centerer.medians[j];
			}
		}
		
		return new Array2DRowRealMatrix(data, false);
	}

	@Override
	public RobustScaler copy() {
		return new RobustScaler(this);
	}

	@Override
	public RobustScaler fit(RealMatrix X) {
		synchronized(fitLock) {
			this.centerer = new MedianCenterer().fit(X);
			
			// Get percentile
			final int n = X.getColumnDimension();
			double[][] transpose = X.transpose().getData();
			
			// top row will be 25th, bottom 75
			double[][] quantiles_25_75 = new double[2][n];
			
			// Quantile engine
			DescriptiveStatistics stats;
			for(int j = 0; j < n; j++) {
				stats = new DescriptiveStatistics();
				
				for(int i = 0; i < transpose[j].length; i++) {
					stats.addValue(transpose[j][i]);
				}
				
				quantiles_25_75[0][j] = stats.getPercentile(25);
				quantiles_25_75[0][j] = stats.getPercentile(75);
			}
			
			// set the scale
			this.scale = VecUtils.subtract(quantiles_25_75[1], quantiles_25_75[0]);
			
			// If we have a constant value, we might get zeroes in the scale:
			for(int i = 0; i < scale.length; i++) {
				if(scale[i] == 0) {
					scale[i] = 1.0;
				}
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
		
		// Dim mismatch will happen on the median side
		double[][] centered = centerer.transform(data);
		
		// Scale:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				centered[i][j] /= scale[j];
			}
		}
		
		return centered;
	}
}
