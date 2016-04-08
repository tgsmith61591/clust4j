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
package com.clust4j.kernel;

import com.clust4j.utils.VecUtils;

/**
 * The Histogram Intersection Kernel is also known as the 
 * Min Kernel and has been proven useful in image classification.
 *
 * @see <a href="http://crsouza.blogspot.com/2010/03/kernel-functions-for-machine-learning.html">Souza, Cesar R. -- Kernel Functions for Machine Learning Applications.</a>
 * @author Taylor G Smith
 */
public class MinKernel extends Kernel {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6559633676695313938L;

	public MinKernel() {
		super();
	}

	@Override
	public double getSimilarity(double[] a, double[] b) {
		return VecUtils.sum(VecUtils.pmin(a, b));
	}

	@Override
	public String getName() {
		return "Min (Histogram Intersection) Kernel";
	}

}
