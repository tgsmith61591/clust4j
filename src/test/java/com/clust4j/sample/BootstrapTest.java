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
package com.clust4j.sample;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.utils.MatrixFormatter;

public class BootstrapTest {
	final static MatrixFormatter formatter = new MatrixFormatter();
	
	static Array2DRowRealMatrix toMatrix(final double[][] data) {
		return new Array2DRowRealMatrix(data, false);
	}
	
	public static void printMatrix(final double[][] d) {
		System.out.println(formatter.format(toMatrix(d)));
	}
	
	public static void printArray(final double[] d) {
		System.out.println(Arrays.toString(d));
	}

	@Test
	public void testUniform() {
		final double[][] data = new double[][]{
			new double[]{1,2,3,4,5},
			new double[]{9,8,7,6,5},
			new double[]{1,0,2,9,3}
		};
		
		final double[][] sampled = Bootstrapper.BASIC.sample(data, 8);
		assertTrue(sampled.length == 8);
	}
	
	@Test
	public void testSmooth() {
		final double[][] data = new double[][]{
			new double[]{1,2,3,4,5},
			new double[]{9,8,7,6,5},
			new double[]{1,0,2,9,3},
			new double[]{90,18,2,0.4,2}
		};
		
		final double[][] sampled = Bootstrapper.SMOOTH.sample(data, 8);
		printMatrix(sampled);
		assertTrue(sampled.length == 8);
	}

}
