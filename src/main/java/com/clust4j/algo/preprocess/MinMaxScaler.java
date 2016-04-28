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

public class MinMaxScaler extends Transformer {
	private static final long serialVersionUID = 2028554388465841136L;
	public static final int DEF_MIN = 0;
	public static final int DEF_MAX = 1;
	
	volatile double[] mins;
	volatile double[] maxes;
	
	private final int min, max;
	
	private MinMaxScaler(MinMaxScaler instance) {
		this.mins = VecUtils.copy(instance.mins);
		this.maxes= VecUtils.copy(instance.maxes);
		this.min  = instance.min;
		this.max  = instance.max;
	}
	
	public MinMaxScaler() {
		this(DEF_MIN, DEF_MAX);
	}
	
	public MinMaxScaler(int min, int max) {
		if(min >= max)
			throw new IllegalStateException("RANGE_MIN ("+min+
					") must be lower than RANGE_MAX ("+max+")");
		
		this.min = min;
		this.max = max;
	}
	
	
	@Override
	protected void checkFit() {
		if(null == mins)
			throw new ModelNotFitException("model not yet fit");
	}
	
	@Override
	public MinMaxScaler copy() {
		return new MinMaxScaler(this);
	}

	@Override
	public MinMaxScaler fit(RealMatrix X) {
		synchronized(fitLock) {
			final int m = X.getRowDimension();
			final int n = X.getColumnDimension();
			
			this.mins = new double[n];
			this.maxes= new double[n];
			double[][] data = X.getData();
			
			for(int col = 0; col < n; col++) {
				double mn = Double.POSITIVE_INFINITY, mx = Double.NEGATIVE_INFINITY;
				
				for(int row = 0; row < m; row++) {
					mn = FastMath.min(mn, data[row][col]);
					mx = FastMath.max(mx, data[row][col]);
				}
				
				this.mins[col] = mn;
				this.maxes[col]= mx;
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
		
		if(n != mins.length)
			throw new DimensionMismatchException(n, mins.length);

		double[][] X = new double[m][n];
		// second pass, subtract to center:
		for(int j = 0; j < n; j++) {
			double mn = mins[j];
			double rng = maxes[j] - mn;
			
			for(int i = 0; i < m; i++) {
				X[i][j] = ((data[i][j] - mn) / rng) * (max - min) + min;
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
		
		if(n != mins.length)
			throw new DimensionMismatchException(n, mins.length);
		
		double rng, mn;
		for(int j = 0; j < n; j++) {
			mn = mins[j];
			rng= maxes[j] - mn;
			
			for(int i = 0; i < m; i++) {
				data[i][j] -= min; // First subtract the min
				data[i][j] /= (max - min); // then divide over max - min
				data[i][j] *= rng; // multiply back by the range
				data[i][j] += mn; // finally add the mn back
			}
		}
		
		return new Array2DRowRealMatrix(data, false);
	}
}
