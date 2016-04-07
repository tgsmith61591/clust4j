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

@Deprecated
final public class DistributedNaNCount extends ReduceTaskOperator<Integer> {
	private static final long serialVersionUID = 5031788548523204436L;

	private DistributedNaNCount(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected DistributedNaNCount newInstance(double[] array, int low, int high) {
		return new DistributedNaNCount(array, low, high);
	}

	@Override
	protected Integer joinSides(Integer left, Integer right) {
		return left + right; // Sum of counts
	}

	@Override
	protected Integer operate(int lo, int hi) {
		int sum = 0;
        for(int i=lo; i < hi; ++i) 
            if(Double.isNaN(array[i]))
            	sum++;
        return sum;
	}
	
	public static int operate(double[] array) {
		return getThreadPool().invoke(new DistributedNaNCount(array,0,array.length));
	}

    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}