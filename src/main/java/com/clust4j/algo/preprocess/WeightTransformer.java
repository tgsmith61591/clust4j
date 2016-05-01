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

import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

/**
 * The weight transformer takes a vector of weights as a constructor argument,
 * and applies the weights to incoming data multiplicatively column-wise. This
 * transformer behaves differently than others, in that the {@link #fit(RealMatrix)}
 * method does not change the state of the transformer, but merely allows it to
 * conform to the {@link PreProcessor} API
 * @author Taylor G Smith
 */
public class WeightTransformer extends Transformer {
	private static final long serialVersionUID = -4256256213984769852L;
	final static double Inf = Double.POSITIVE_INFINITY;
	final double[] weights;
	final int n;
	
	
	private WeightTransformer(WeightTransformer wt) {
		this.weights = VecUtils.copy(wt.weights);
		this.n = wt.n;
	}
	
	public WeightTransformer(double[] weights) {
		this.weights = VecUtils.copy(weights);
		this.n = weights.length;
	}
	
	

	@Override
	protected void checkFit() {
		; // will always be fit, per constructor...
	}

	/**
	 * Inverse transform the incoming data. If the corresponding weight is 0.0,
	 * will coerce the column to positive infinity rather than NaN.
	 */
	@Override
	public RealMatrix inverseTransform(RealMatrix data) {
		checkFit();
		
		final int m = data.getRowDimension();
		if(data.getColumnDimension() != n)
			throw new DimensionMismatchException(n, data.getColumnDimension());

		double[][] X = data.getData();
		double weight, val;
		for(int j = 0; j < n; j++) {
			weight = weights[j];
			
			for(int i = 0; i < m; i++) {
				// sometimes, weight can be 0.0 if the user is masochistic...
				val = X[i][j] / weight;
				X[i][j] = Double.isNaN(val) ? Inf : val;
			}
		}
		
		// assign -- already copied in getData()
		return new Array2DRowRealMatrix(X, false);
	}

	@Override
	public WeightTransformer copy() {
		return new WeightTransformer(this);
	}

	@Override
	public WeightTransformer fit(RealMatrix X) {
		synchronized(fitLock) {
			// Only enforce this to prevent accidental exceptions later if the user
			// tries a fit(X).transform(X) and later gets a dim mismatch...
			if(X.getColumnDimension() != n)
				throw new DimensionMismatchException(n, X.getColumnDimension());
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
		if(data[0].length != n)
			throw new DimensionMismatchException(n, data[0].length);

		double[][] X = new double[m][n];
		// mult to weight:
		for(int j = 0; j < n; j++) {
			for(int i = 0; i < m; i++) {
				X[i][j] = data[i][j] * weights[j];
			}
		}
		
		// assign
		return X;
	}

}
