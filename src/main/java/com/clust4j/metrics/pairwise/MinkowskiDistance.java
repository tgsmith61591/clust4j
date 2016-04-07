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
package com.clust4j.metrics.pairwise;

import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.VecUtils;

public class MinkowskiDistance implements DistanceMetric {
	private static final long serialVersionUID = 6206826797866732365L;
	final private double p;
	
	public MinkowskiDistance(final double p) {
		if(p < 1)
			throw new IllegalArgumentException("p cannot be less than 1");
		this.p = p;
	}

	@Override
	public double getDistance(double[] a, double[] b) {
		return partialDistanceToDistance(getPartialDistance(a, b));
	}
	
	@Override
	final public double getP() {
		return p;
	}
	
	@Override
	public double getPartialDistance(final double[] a, final double[] b) {
		VecUtils.checkDims(a,b);
		
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			// Don't use math.abs -- too expensive
			double diff = a[i] - b[i];
			sum += FastMath.pow(FastMath.abs(diff), p);
		}
		
		return sum;
	}
	
	@Override
	public double partialDistanceToDistance(double d) {
		return FastMath.pow(d, 1.0/p);
	}
	
	@Override
	public double distanceToPartialDistance(double d) {
		return FastMath.pow(d, this.p);
	}
	
	@Override
	public String getName() {
		return "Minkowski";
	}
	
	@Override
	public String toString() {
		return getName();
	}
}
