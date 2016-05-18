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
import org.apache.commons.math3.util.FastMath;

public class YeoJohnsonTransformer extends BoxCoxTransformer {
	private static final long serialVersionUID = -6918706472624701296L;
	
	
	
	private YeoJohnsonTransformer(YeoJohnsonTransformer instance) {
		super(instance);
	}
	
	public YeoJohnsonTransformer() {
		super();
	}
	
	public YeoJohnsonTransformer(double min_lam, double max_lam) {
		super(min_lam, max_lam);
	}
	
	
	
	@Override
	public YeoJohnsonTransformer copy() {
		return new YeoJohnsonTransformer(this);
	}
	
	@Override
	protected double[] estimateShifts(double[][] x) {
		// shift should equal zero for YJ
		return new double[x.length];
	}
	
	@Override
	public YeoJohnsonTransformer fit(RealMatrix X) {
		super.fit(X);
		return this;
	}
	
	/**
	 * Inverse transform your matrix.
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
			for(int i = 0; i < m; i++) {
				x[i][j] = yjInvTransSingle(x[i][j], this.lambdas[j]);
			}
		}
		
		// Implicit copy in the getData()
		return new Array2DRowRealMatrix(x, false);
	}
	
	private static double yjInvTransSingle(double x, double lam) {
		/*
		 * This is where it gets messy, but we can theorize that
    	 * if the x is < 0 and the lambda meets the appropriate conditions,
    	 * that the x was sub-zero to begin with
		 */
		if(x >= 0) {
			// Case 1: x >= 0 and lambda is not 0
			if(!nearZero(lam)) {
				x *= lam;
				x += 1;
				x = FastMath.pow(x, 1.0 / lam);
				return x - 1;
			}
			
			// Case 2: x >= 0 and lambda is 0
			return FastMath.exp(x) - 1;
		} else {
			// Case 3: lambda does not equal 2
			if(lam != 2.0) {
				x *= -(2.0 - lam);
	            x += 1;
	            x = FastMath.pow(x, 1.0 / (2.0 - lam));
	            x -= 1;
	            return -x;
			}
			
			// Case 4: lambda equals 2
			return -(FastMath.exp(-x) - 1);
		}
	}
	
	private static boolean nearZero(double a) {
		return FastMath.abs(a) <= zero;
	}
	
	/**
	 * Shift and transform the feature
	 * @param y
	 * @param lambda
	 * @return
	 */
	@Override
	double lambdaTransform(double y, double lambda) {
		if(lambda != zero && y >= 0.0) {
			return (FastMath.pow(y + 1, lambda) - 1) / lambda;
		} else if(lambda == zero && y >= 0.0) {
			return FastMath.log(y + 1);
		} else if(lambda != 2 && y < 0.0) {
			return -(FastMath.pow(-y + 1, 2.0 - lambda) - 1) / (2.0 - lambda);
		} else {
			return -FastMath.log(-y + 1);
		}
	}
}
