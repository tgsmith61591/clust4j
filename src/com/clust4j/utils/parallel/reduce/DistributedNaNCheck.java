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
 * A class for distributed NaN checks
 * @author Taylor G Smith
 */
@Deprecated
public class DistributedNaNCheck extends ReduceTaskOperator<Boolean> {
	private static final long serialVersionUID = -4107497709587691394L;

	private DistributedNaNCheck(final double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected DistributedNaNCheck newInstance(double[] array, int low, int high) {
		return new DistributedNaNCheck(array, low, high);
	}

	@Override
	protected Boolean joinSides(Boolean left, Boolean right) {
		return left || right;
	}

	@Override
	protected Boolean operate(int lo, int hi) {
		for(int i=lo; i < hi; ++i)
            if(Double.isNaN(array[i]))
        		return true;
        return false;
	}
	
	public static boolean operate(final double[] array) {
		return getThreadPool().invoke(new DistributedNaNCheck(array,0,array.length));
	}

    
    @Override void checkDims(double[] v) {
    	VecUtils.checkDimsPermitEmpty(v);
    }
}
