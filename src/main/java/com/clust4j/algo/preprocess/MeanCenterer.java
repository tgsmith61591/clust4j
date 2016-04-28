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

import com.clust4j.except.ModelNotFitException;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class MeanCenterer extends Transformer {
	private static final long serialVersionUID = 2028554388465841136L;
	volatile double[] means;
	
	private MeanCenterer(MeanCenterer instance) {
		this.means = VecUtils.copy(instance.means);
	}
	
	public MeanCenterer() {
	}
	
	
	@Override
	protected void checkFit() {
		if(null == means)
			throw new ModelNotFitException("model not yet fit");
	}
	
	@Override
	public MeanCenterer copy() {
		return new MeanCenterer(this);
	}

	@Override
	public MeanCenterer fit(RealMatrix data) {
		synchronized(fitLock) {
			final int m = data.getRowDimension();
			final int n = data.getColumnDimension();

			// need to mean center...
			this.means = new double[n];
			final double[][] y = data.getData();
			
			// First pass, compute mean...
			for(int j = 0; j < n; j++) {
				for(int i = 0; i < m; i++) {
					means[j] += y[i][j];
					
					// if last:
					if(i == m - 1) {
						means[j] /= (double)m;
					}
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
		
		if(n != means.length)
			throw new DimensionMismatchException(n, means.length);

		double[][] X = new double[m][n];
		// second pass, subtract to center:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = data[i][j] - means[j];
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
				data[i][j] += means[j];
			}
		}
		
		return new Array2DRowRealMatrix(data, false);
	}
}
