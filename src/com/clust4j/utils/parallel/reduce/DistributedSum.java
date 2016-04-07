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
package com.clust4j.utils.parallel.reduce;

import com.clust4j.utils.VecUtils;

/**
 * A class for distributed summing of vectors
 * @author Taylor G Smith
 */
@Deprecated
final public class DistributedSum extends ReduceTaskOperator<Double> {
	private static final long serialVersionUID = -6086182277529660733L;

    private DistributedSum(final double[] arr, int lo, int hi) {
        super(arr, lo, hi);
    }

	@Override
	protected DistributedSum newInstance(double[] array, int low, int high) {
		return new DistributedSum(array, low, high);
	}

	@Override
	protected Double joinSides(Double left, Double right) {
		return left + right; // Sum
	}

	@Override
	protected Double operate(int lo, int hi) {
		double sum = 0;
		for(int i = lo; i < hi; i++)
			sum += array[i];
		return sum;
	}

    public static double operate(final double[] array) {
    	if(array.length == 0)
    		return 0;
        return getThreadPool().invoke(new DistributedSum(array,0,array.length));
    }
    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}