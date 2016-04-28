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
package com.clust4j.utils;

import static org.junit.Assert.*;

import org.junit.Test;

import com.clust4j.utils.TableFormatter.ColumnAlignment;

public class FormatterTests {
	final MatrixFormatter formatter = new MatrixFormatter();

	@Test
	public void testMatrixUniform() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5,6}
		};
		
		formatter.format(d);
		assertTrue(true); // get rid of import warning...
	}
	
	@Test
	public void testMatrixNonUniform() {
		double[][] d = new double[][]{
			new double[]{1,2,3},
			new double[]{4,5}
		};
		
		formatter.format(d);
	}

	@Test
	public void testMatrixEmptyCols() {
		double[][] d = new double[][]{
			new double[]{},
			new double[]{}
		};
		
		formatter.format(d);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testMatrixEmptyRows() {
		double[][] d = new double[][]{ };
		formatter.format(d);
	}
	
	@Test
	public void testIntHead() {
		int[][] i = new int[][]{
			new int[]{1,2,3},
			new int[]{1,2,3},
			new int[]{1,2,3}
		};
		
		final MatrixFormatter left_align = new MatrixFormatter(ColumnAlignment.LEFT);
		// show just top two, ensure doesn't cause index out of bounds or anything
		System.out.println(left_align.format(i, 2));
		// try it with one too many, make sure it only shows the head
		System.out.println(left_align.format(i, 4));
		
		assertTrue(left_align.getAlignment() == ColumnAlignment.LEFT);
		left_align.toggleAlignment();
		assertTrue(left_align.getAlignment() == ColumnAlignment.RIGHT);
		
		assertTrue(left_align.prefix.isEmpty());
		assertTrue(left_align.suffix.isEmpty());
		assertTrue(left_align.rowPrefix.isEmpty());
		assertTrue(left_align.rowSuffix.isEmpty());
		assertTrue(left_align.columnSeparator.isEmpty());
		assertTrue(left_align.rowSeparator.equals(System.getProperty("line.separator")));
		assertTrue(left_align.getWhitespace() == 4);
		assertNotNull(left_align.format);
		
	}
}
