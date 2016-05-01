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
	
	public YeoJohnsonTransformer(double min_lam, double max_lam, double inc) {
		super(min_lam, max_lam, inc);
	}
	
	
	
	@Override
	public YeoJohnsonTransformer copy() {
		return new YeoJohnsonTransformer(this);
	}
	
	@Override
	public void estimateShifts(double[][] x) {
		// shift should equal zero for YJ
		this.shift = new double[x.length];
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
		/*
		 * The YeoJohnson suffers in this regard, as there
		 * is no good way to determine exactly which transformation
		 * to reverse just given the lambda values... we have to
		 * admit defeat here, in a way, and just call the super
		 * method.
		 */
		return super.inverseTransform(X);
	}
	
	/**
	 * Shift and transform the feature
	 * @param y
	 * @param shift
	 * @param lambda
	 * @return
	 */
	@Override
	double lambdaTransform(double y, double shift, double lambda) {
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
