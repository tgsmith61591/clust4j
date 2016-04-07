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
package com.clust4j.utils.parallel.map;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

@Deprecated
public class DistributedLog extends MapTaskOperator {
	private static final long serialVersionUID = -3885390722365779996L;

	DistributedLog(double[] arr, int lo, int hi) {
		super(arr, lo, hi);
	}

	@Override
	protected double operate(double a) {
		return FastMath.log(a);
	}

	@Override
	protected MapTaskOperator newInstance(double[] array, int low, int high) {
		return new DistributedLog(array, low, high);
	}
	
	public static double[] operate(final double[] array) {
		VecUtils.checkDimsPermitEmpty(array);
		return getThreadPool().invoke(new DistributedLog(array, 0, array.length));
    }
}
