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

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MedianCenterer extends Transformer {
	private static final long serialVersionUID = -5983524673626323084L;
	volatile protected double[] medians;
	
	
	private MedianCenterer(MedianCenterer mc) {
		this.medians = VecUtils.copy(mc.medians);
	}
	
	public MedianCenterer() {
	}
	
	

	@Override
	protected void checkFit() {
		if(null == medians)
			throw new ModelNotFitException("model not yet fit");
	}

	@Override
	public RealMatrix inverseTransform(RealMatrix X) {
		checkFit();
		
		// This effectively copies, so no need to do a copy later
		double[][] data = X.getData();
		final int m = data.length;
		final int n = data[0].length;
		
		if(n != medians.length)
			throw new DimensionMismatchException(n, medians.length);
		
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				data[i][j] += medians[j];
			}
		}
		
		return new Array2DRowRealMatrix(data, false);
	}

	@Override
	public MedianCenterer copy() {
		return new MedianCenterer(this);
	}

	@Override
	public MedianCenterer fit(RealMatrix X) {
		synchronized(fitLock) {
			final int n = X.getColumnDimension();

			// need to mean center...
			this.medians = new double[n];
			final double[][] y = X.transpose().getData();
			
			// First pass, compute median...
			for(int j = 0; j < n; j++) {
				this.medians[j] = VecUtils.median(y[j]);
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
		
		if(n != medians.length)
			throw new DimensionMismatchException(n, medians.length);

		double[][] X = new double[m][n];
		// subtract to center:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = data[i][j] - medians[j];
			}
		}
		
		// assign
		return X;
	}
}
