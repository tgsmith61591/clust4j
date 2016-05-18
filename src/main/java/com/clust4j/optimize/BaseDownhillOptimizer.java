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

package com.clust4j.optimize;


/**
 * Given a user-suggested lower and upper bounds to a bracket
 * for an optimization function, identify the argmin to an
 * {@link OptimizableCaller} objective function. Precedence
 * will be given to the user-supplied min/max, but a result within
 * the boundaries is not guaranteed.
 * 
 * @author Taylor G Smith
 */
public abstract class BaseDownhillOptimizer {
	final OptimizableCaller optimizer;
	final Bracket bracket;
	final static double tol = 1.48e-8;
	final static int maxIter = 500;
	
	private boolean hasOptimized = false;
	private double optimalValue = Double.NaN;
	
	public BaseDownhillOptimizer(OptimizableCaller callable) {
		this(callable, 0.0, 1.0);
	}
	
	public BaseDownhillOptimizer(OptimizableCaller callable, double min, double max) {
		this.optimizer = callable;
		
		// assert min less than max
		if(min >= max)
			throw new IllegalArgumentException("min must be less than max");
		
		// do bracket search
		this.bracket = new Bracket(callable, min, max);
	}
	
	
	final public double optimize() {
		if(hasOptimized) {
			return optimalValue;
		} else {
			this.hasOptimized = true;
			return this.optimalValue = optimizeImplementation();
		}
	}
	
	abstract protected double optimizeImplementation();
	abstract public int getNumFunctionCalls();
	abstract public double getFunctionResult();
}
