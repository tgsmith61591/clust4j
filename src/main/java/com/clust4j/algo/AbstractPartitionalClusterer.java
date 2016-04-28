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
package com.clust4j.algo;

import org.apache.commons.math3.linear.RealMatrix;

public abstract class AbstractPartitionalClusterer extends AbstractClusterer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 8489725366968682469L;
	/**
	 * The number of clusters to find. This field is not final, as in
	 * some corner cases, the algorithm will modify k for convergence.
	 */
	protected int k;
	
	public AbstractPartitionalClusterer(
			RealMatrix data, 
			BaseClustererParameters planner,
			final int k) 
	{
		super(data, planner);
		
		if(k < 1)
			error(new IllegalArgumentException("k must exceed 0"));
		if(k > data.getRowDimension())
			error(new IllegalArgumentException("k exceeds number of records"));
		
		this.k = this.singular_value ? 1 : k;
		if(this.singular_value && k!=1) {
			warn("coerced k to 1 due to equality of all elements in input matrix");
		}
	} // End constructor
	
	public int getK() {
		return k;
	}
}
